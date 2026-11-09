from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


_DEFAULT_LEARNING_RATE = {"pytorch": 0.0001, "gensim": 0.025}


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

    embedding_model : {"skipgram", "cbow"}, default "skipgram"
        Word2Vec variant used for embedding training. Multi-GPU CBOW
        requires ``backend="gensim"``.

    epochs : int, default 5
        Number of training epochs.

    batch_size : int or None, default None
        Training batch size of the PyTorch backend. If None, a single-GPU run
        uses 1/100 of the training pairs (or, with ``tune_batch_size=True``,
        the tuner's result) and a multi-GPU run uses 512. An explicit value is
        always used as given. Not used by ``backend="gensim"``.

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
        Learning rate. With ``backend="pytorch"`` it is the optimiser's
        learning rate; with ``backend="gensim"`` it is gensim's initial
        ``alpha``. If None, it is set to the backend's default: 0.0001 for
        ``"pytorch"`` and 0.025 (gensim's own default) for ``"gensim"``.

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

    cpu_count : int, default 4
        Number of gensim worker threads with ``backend="gensim"``. Not used
        by the PyTorch backend.

    tune_batch_size : bool, default True
        Run Lightning's batch-size tuner (power scaling, starting from 1/200
        of the training pairs) before single-GPU PyTorch training when
        ``batch_size`` is None. Not used by multi-GPU training or
        ``backend="gensim"``.

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
    embedding_model: Literal["skipgram", "cbow"] = "skipgram"
    epochs: int = Field(default=5, gt=0)
    batch_size: Optional[int] = Field(default=None, gt=0)
    vector_size: int = Field(default=256, gt=0)
    window_size: int = Field(default=5, gt=1)
    min_count: int = Field(default=1, ge=0)
    negative_samples: int = Field(default=5, ge=0)
    learning_rate: Optional[float] = Field(default=None, gt=0)
    backend: Literal["pytorch", "gensim"] = "pytorch"
    # library settings
    random_state: int = Field(default=42, ge=0)
    reproducible: bool = False
    multi_gpu: bool = False
    generate_artifact: bool = False
    cpu_count: int = Field(default=4, gt=0)
    tune_batch_size: bool = True
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
        # the two trainers need very different learning rates: gensim's
        # `alpha` decays linearly from 0.025, the PyTorch optimiser uses 1e-4
        if self.learning_rate is None:
            self.learning_rate = _DEFAULT_LEARNING_RATE[self.backend]
        return self

    @model_validator(mode="after")
    def _check_multi_gpu_cbow(self):
        if self.multi_gpu and self.backend == "pytorch" and self.embedding_model == "cbow":
            # its training pairs lose the context windows (see `_cbow_pairs`)
            raise ValueError(
                "multi-GPU CBOW is not supported with backend='pytorch'; use backend='gensim' or a single GPU"
            )
        return self
