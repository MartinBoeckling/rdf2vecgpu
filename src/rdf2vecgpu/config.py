import os
import warnings
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# the PyTorch trainers sum the loss over a batch and use SGD, so the learning
# rate is a step size per example like gensim's alpha; CBOW on BFS walks still
# uses Adam on a mean loss
_DEFAULT_LEARNING_RATE = {
    ("pytorch", "skipgram"): 0.025,
    ("pytorch", "cbow"): 0.025,
    ("gensim", "skipgram"): 0.025,
    ("gensim", "cbow"): 0.025,
    ("pytorch", "order_aware_skipgram"): 0.025,
    ("gensim", "order_aware_skipgram"): 0.025,  # rejected below; keeps the lookup total
}
_BFS_CBOW_LEARNING_RATE = 0.0001
DEFAULT_BATCH_SIZE = 32_768
# training uses at most 1/MIN_STEPS_PER_EPOCH of the pairs per step, so a
# small corpus still gets this many SGD steps per epoch
MIN_STEPS_PER_EPOCH = 100


def _available_cores() -> int:
    """CPU cores this process may run on (respects affinity and cgroup
    pinning, e.g. a SLURM allocation), at least 1."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not on Linux
        return max(1, os.cpu_count() or 1)


class RDF2VecConfig(BaseModel):
    """
    Configuration for GPU-accelerated RDF2Vec.

    A Pydantic model that holds every hyperparameter of the pipeline:

    - walk generation
    - the Word2Vec model and its training backend
    - training behaviour (epochs, batch size, seeding)
    - execution (single GPU or multi-GPU)
    - literal handling
    - experiment tracking

    Values are validated on construction, so an out-of-range value (for
    example ``walk_depth=0``) raises a ``pydantic.ValidationError``.

    Parameters
    ----------
    walk_strategy : {"random", "bfs"}, default "random"
        Strategy used to generate walks from the knowledge graph. ``"bfs"``
        runs one breadth-first search per start vertex; its cost per vertex
        grows with the size of the graph, so it suits small graphs only.
        ``backend="gensim"`` supports ``"random"`` only.

    walk_depth : int, default 4
        Maximum number of hops of each walk.

    walk_number : int, default 100
        Number of random walks started per start vertex. Not used by
        ``"bfs"``.

    walk_weighted : bool, default False
        If True, random walks pick each out-edge with probability
        proportional to its weight (cuGraph ``biased_random_walks``). The
        input data must contain a ``weights`` column. Weights must be finite
        and non-negative, and at least one must be positive. A weight-0 edge
        is never walked; a vertex whose out-edges all weigh 0 ends the walk.
        BFS walks ignore weights.

    embedding_model : {"skipgram", "cbow", "order_aware_skipgram"}, default "skipgram"
        Word2Vec variant used for embedding training, all with negative
        sampling. ``"order_aware_skipgram"`` (RDF2Vec_oa) learns a separate
        output embedding per position in the context window, so it also
        captures where a token occurs relative to another; it needs
        ``backend="pytorch"`` and random walks, and its output table has
        ``2 * window_size`` times the rows of skip-gram's. Multi-GPU CBOW on
        BFS walks requires ``backend="gensim"``.

    epochs : int, default 5
        Number of training epochs.

    batch_size : int or None, default None
        Number of (center, context) pairs per training step of the PyTorch
        backend; None means 32768. Training uses at most 1/100 of the
        training pairs (per GPU) per step, so every epoch has at least 100
        steps; a larger value is reduced, with a warning when set explicitly.
        Skip-gram on random walks fills each step with whole walks, so a step
        holds about ``batch_size`` pairs (fewer when walks end early or
        ``sample`` drops tokens). Larger batches train faster with slightly
        lower embedding quality. Not used by ``backend="gensim"``.

    vector_size : int, default 256
        Dimensionality of the output embeddings.

    window_size : int, default 5
        Context window size for Word2Vec.

    min_count : int, default 1
        Minimum token frequency in the walks for a token to take part in
        training. The vocabulary and the output table always cover every
        token. Tokens below the threshold keep their random initial vector
        with ``backend="pytorch"`` and get a zero vector with
        ``backend="gensim"``.

    negative_samples : int, default 5
        Number of negative examples for negative sampling.

    learning_rate : float or None, default None
        Initial learning rate: a step size per training example that decays
        linearly to ``min_learning_rate`` over training (gensim's ``alpha``),
        for both backends and models. If None, it is set to 0.025 (gensim's
        own default); PyTorch CBOW on BFS walks, which still uses Adam, gets
        0.0001.

    min_learning_rate : float, default 0.0001
        Learning rate at the end of training, after the linear decay
        (gensim's ``min_alpha``). Not used by PyTorch CBOW.

    sample : float, default 0.001
        Threshold for dropping occurrences of frequent tokens from the walks
        during training, as gensim's ``sample``: a token that makes up a
        fraction ``f`` of the walks keeps each occurrence with probability
        ``(sqrt(f / sample) + 1) * sample / f``. 0 keeps everything (as
        jRDF2Vec does by default). Used by skip-gram on random walks with
        ``backend="pytorch"`` and by ``backend="gensim"``.

    shrink_window : bool, default False
        Draw each centre token's context window uniformly from
        1..``window_size``, as word2vec and gensim do ("reduced windows"),
        instead of always using the full window, for the PyTorch skip-gram,
        order-aware skip-gram and CBOW models. Near tokens then weigh more
        than far ones, and a walk gives 33-40% fewer pairs (batches are
        sized by the expected count). ``backend="gensim"`` always shrinks.

    grad_clip_value : float, default 1.0
        Per-component clip of the summed batch gradient for the PyTorch
        skip-gram, order-aware skip-gram and CBOW models, as a guard against
        overflowing steps for very frequent tokens. 0 turns it off. Values
        well below 1 slow the learning of frequent tokens and lower embedding
        quality. Not used by ``backend="gensim"``.

    backend : {"pytorch", "gensim"}, default "pytorch"
        Word2Vec trainer. ``"pytorch"`` trains on the GPU(s) with PyTorch
        Lightning. ``"gensim"`` trains on the CPU with gensim (install the
        ``rdf2vecgpu[gensim]`` extra); the walks are still generated on the
        GPU.

    random_state : int, default 42
        Seed for walk sampling and, with ``reproducible=True``, for model
        training.

    reproducible : bool, default False
        If True, seed Python, NumPy and PyTorch with ``random_state`` before
        training (Lightning ``seed_everything``). It does not switch on
        deterministic CUDA kernels, so GPU runs can still differ slightly.

    multi_gpu : bool, default False
        If True, read the data, build the graph and generate walks with
        Dask-cuDF across all GPUs of a Dask cluster, and train on all of them.
        Requires a Dask ``client`` passed to ``GPU_RDF2Vec``.

    generate_artifact : bool, default False
        If True, ``transform()`` also writes the embeddings to
        ``./vector/embeddings_<embedding_model>.parquet`` and the vocabulary
        to ``./vector/word2idx.parquet``.

    cpu_count : int, default: the CPU cores available to the process
        Number of gensim worker threads with ``backend="gensim"``. Not used
        by the PyTorch backend.

    tune_batch_size : bool, default False
        Deprecated and ignored; it will be removed in 0.7.0. The batch-size
        tuner searched for the largest batch that fits in GPU memory, which is
        no longer the limit; set ``batch_size`` instead.

    num_nodes : int, default 1
        Number of nodes for multi-GPU PyTorch training (Lightning
        ``num_nodes``). ``backend="gensim"`` requires a single node.

    tracker : {"none", "mlflow", "wandb"}, default "none"
        Experiment tracking backend. ``"mlflow"`` and ``"wandb"`` need the
        ``rdf2vecgpu[mlflow]`` or ``rdf2vecgpu[wandb]`` extra.

    tracker_kwargs : dict or None, default None
        Keyword arguments for the tracker, keyed by backend name, for example
        ``{"mlflow": {"experiment": "rdf2vec", "tracking_uri": "file:./mlruns"}}`` or
        ``{"wandb": {"project": "rdf2vec"}}``.

    tracker_run_name : str or None, default None
        Run name passed to the tracker.

    literal_predicates : list[str] or None, default None
        Predicates that identify literal (numeric) edges. When set, edges
        with these predicates are handled according to ``literal_strategy``.
        Matching is exact against the loaded data; the N-Triples reader
        strips the angle brackets from IRIs, so give IRIs without ``<>``
        for ``.nt``/``.nq`` input.

    literal_strategy : {"drop", "bin"}, default "drop"
        How to handle literal edges. ``"drop"`` removes them from the graph
        (pyRDF2Vec default). ``"bin"`` discretizes the object values into
        bin tokens so the edge stays in the graph.

    literal_n_bins : int, default 5
        Number of bins when ``literal_strategy="bin"``.

    literal_bin_strategy : {"quantile", "uniform"}, default "quantile"
        Binning method. ``"quantile"`` creates equal-frequency bins (robust
        to skew). ``"uniform"`` creates equal-width bins.
    """

    # walk parameter settings
    walk_strategy: Literal["random", "bfs"] = "random"
    walk_depth: int = Field(default=4, gt=0)
    walk_number: int = Field(default=100, gt=0)
    walk_weighted: bool = False
    # embedding parameter settings
    embedding_model: Literal["skipgram", "cbow", "order_aware_skipgram"] = "skipgram"
    epochs: int = Field(default=5, gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    vector_size: int = Field(default=256, gt=0)
    window_size: int = Field(default=5, gt=1)
    min_count: int = Field(default=1, ge=0)
    negative_samples: int = Field(default=5, ge=0)
    learning_rate: Optional[float] = Field(default=None, gt=0)
    min_learning_rate: float = Field(default=0.0001, gt=0)
    sample: float = Field(default=0.001, ge=0)
    shrink_window: bool = False
    grad_clip_value: float = Field(default=1.0, ge=0)
    backend: Literal["pytorch", "gensim"] = "pytorch"
    # library settings
    random_state: int = Field(default=42, ge=0)
    reproducible: bool = False
    multi_gpu: bool = False
    generate_artifact: bool = False
    cpu_count: int = Field(default_factory=lambda: _available_cores(), gt=0)
    tune_batch_size: bool = False
    num_nodes: int = Field(default=1, gt=0)
    tracker: Literal["mlflow", "wandb", "none"] = "none"
    tracker_kwargs: Optional[dict] = None
    tracker_run_name: Optional[str] = None
    # literal handling settings
    literal_predicates: Optional[list[str]] = None
    literal_strategy: Literal["drop", "bin"] = "drop"
    literal_n_bins: int = Field(default=5, gt=1)
    literal_bin_strategy: Literal["quantile", "uniform"] = "quantile"

    @model_validator(mode="after")
    def _default_learning_rate(self):
        if self.learning_rate is None:
            if self.backend == "pytorch" and self.embedding_model == "cbow" and self.walk_strategy == "bfs":
                self.learning_rate = _BFS_CBOW_LEARNING_RATE
            else:
                self.learning_rate = _DEFAULT_LEARNING_RATE[(self.backend, self.embedding_model)]
        return self

    @model_validator(mode="after")
    def _default_batch_size(self):
        if self.tune_batch_size:
            warnings.warn(
                "tune_batch_size is deprecated and ignored, and will be removed in 0.7.0; "
                f"set batch_size instead (default {DEFAULT_BATCH_SIZE})",
                FutureWarning,
                stacklevel=2,
            )
        if self.batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE
        return self

    @model_validator(mode="after")
    def _check_order_aware(self):
        if self.embedding_model == "order_aware_skipgram":
            if self.backend != "pytorch":
                raise ValueError("embedding_model='order_aware_skipgram' needs backend='pytorch'")
            if self.walk_strategy != "random":
                # BFS walks reach training as pairs, without their positions
                raise ValueError("embedding_model='order_aware_skipgram' needs walk_strategy='random'")
        return self

    @model_validator(mode="after")
    def _check_multi_gpu_cbow(self):
        if (
            self.multi_gpu
            and self.backend == "pytorch"
            and self.embedding_model == "cbow"
            and self.walk_strategy == "bfs"
        ):
            # BFS walks reach CBOW as pairs, which lose the context windows
            # (see `_cbow_pairs`)
            raise ValueError(
                "multi-GPU CBOW on BFS walks is not supported with backend='pytorch'; "
                "use backend='gensim', random walks or a single GPU"
            )
        return self
