from typing import Literal, Optional
from pydantic import BaseModel, Field


class RDF2VecConfig(BaseModel):
    """
    Configuration object for GPU-accelerated RDF2Vec.

    This dataclass centralizes all hyperparameters controlling:
       • walk generation
       • vocabulary construction
       • Word2Vec model architecture
       • training behavior (epochs, batch sizes, reproducibility)
       • execution backend (single GPU vs multi-GPU)
       • artifact export settings

    Parameters
    ----------
    walk_strategy : {"random", "bfs"}, default "random"
        Strategy used to generate walks from the knowledge graph.

    walk_depth : int, default 4
        Maximum depth of each walk.

    walk_number : int, default 100
        Number of walks started per vertex.

    walk_weighted : bool, default False
        If True, use edge weights for biased walk transitions via
        cuGraph's ``biased_random_walks()``. The input data must contain
        a ``"weights"`` column (cuGraph standard name).

    embedding_model : {"skipgram", "cbow"}, default "skipgram"
        Word2Vec variant used for embedding training.

    vector_size : int, default 256
        Dimensionality of the output embeddings.

    window_size : int, default 5
        Context window size for Word2Vec.

    min_count : int, default 1
        Minimum token frequency for inclusion in the vocabulary.

    negative_samples : int, default 5
        Number of negative examples for negative sampling.

    learning_rate : float, default 0.025
        Learning rate used by the optimizer. (For ``optimizer="sparseadam"``
        a much smaller value, ~1e-3–5e-3, is appropriate.)

    optimizer : {"sgd", "sparseadam"}, default "sgd"
        PyTorch skip-gram (walk-sequence path) optimizer. ``"sgd"`` is classic
        word2vec (plain SGD + the ``lr_decay`` schedule; zero adaptive state ⇒
        the optimizer is O(touched rows), and the per-element ``grad_clip_value``
        is what keeps it stable). ``"sparseadam"`` uses ``torch.optim.SparseAdam``
        — adaptive per-parameter step sizes naturally damp the over-aggregated
        hot predicate tokens and lift the rarely-touched tail; costs O(vocab)
        moment state and is single-GPU only (no DDP).

    sample : float, default 0.001
        PyTorch skip-gram (walk-sequence path): frequent-token subsampling
        threshold (gensim's ``sample``; ``0`` disables; ``1e-3`` is the classic
        word2vec value). When set, each token *occurrence* in a walk is kept with
        probability ``min(1, (sqrt(z/sample) + 1) * sample/z)`` where ``z`` is the
        token's corpus frequency — this thins the hot predicate tokens (which
        appear in a large fraction of every walk) at the source. Approximated on
        GPU by masking out skip-gram pairs that touch a subsampled-out occurrence
        (no window-shift), which still achieves the main effect.

    negative_sampling : {"uniform", "unigram"}, default "unigram"
        PyTorch skip-gram (walk-sequence path) negative-sample distribution.
        ``"uniform"`` draws negatives uniformly over the vocab. ``"unigram"`` uses
        the classic word2vec ``f^0.75`` unigram distribution (built from the walk
        corpus token counts) — moderates how often hot predicates appear as
        negatives. Both ``sample`` and ``negative_sampling="unigram"`` require a
        one-pass token-count over the walk parquet.

    min_learning_rate : float, default 0.0001
        Floor of the linear LR decay for the PyTorch skip-gram path.

    lr_decay : {"linear", "none"}, default "linear"
        LR schedule for the PyTorch skip-gram path. "linear" decays
        ``learning_rate`` → ``min_learning_rate`` over the run (gensim's
        alpha schedule); "none" keeps it constant.

    pair_batch_size : int, default 32768
        PyTorch skip-gram (walk-sequence path): number of skip-gram pairs per
        SGD step. Small values stabilise training on RDF2Vec walk corpora
        (a few predicate tokens dominate the token frequency); the default
        32768 is a good GPU-efficient compromise.

    matmul_precision : {"highest", "high", "medium"}, default "high"
        torch.set_float32_matmul_precision setting; "high" enables TF32 on
        A100/A6000 Tensor Cores for the embedding inner-products.

    grad_clip_value : float, default 0.1
        PyTorch skip-gram (walk-sequence path): the gradient clamp magnitude
        (see ``grad_clip_mode``); caps a single SGD step's move. This is what
        keeps SGD training stable on RDF2Vec walk corpora, where a few predicate
        tokens appear in a large fraction of every batch's skip-gram pairs and
        the summed gradient on those rows would otherwise blow up. ``0`` disables
        clipping.

    grad_clip_mode : {"element", "row"}, default "element"
        PyTorch skip-gram (walk-sequence path): how ``grad_clip_value`` clips the
        embedding-table gradients after backward. ``"element"`` clamps each
        gradient component to ``[-grad_clip_value, +grad_clip_value]`` (bounds a
        per-component move at ``learning_rate * grad_clip_value``).  ``"row"``
        rescales each touched row's gradient so its L2 norm is ≤ ``grad_clip_value``
        (a geometry-aware cap on how far a row drifts per step; leaves a row's
        gradient *direction* intact, unlike element clamping).

    accumulate_grad_batches : int, default 1
        Thin pass-through to PyTorch Lightning's gradient accumulation.
        Increasing it enlarges the effective batch and is generally NOT
        recommended for skip-gram (see ``pair_batch_size``).

    epochs : int, default 5
        Number of training epochs.

    batch_size : int or None, default None
        Explicit batch size; if None, Lightning's tuner may pick one.

    tune_batch_size : bool, default False
        Whether to use PyTorch Lightning’s automatic batch size tuning.

    random_state : int, default 42
        Seed for reproducible walk sampling and model initialization.

    reproducible : bool, default True
        If True, enables deterministic modes in PyTorch and CUDA.

    multi_gpu : bool, default False
        If True, enables multi-GPU walk generation and training using Dask.

    cpu_count : int, default 4
        Number of CPU workers used.

    generate_artifact : bool, default False
        If True, persist `word2idx` and embeddings to Parquet files.

    num_nodes : int, default 1
        Number of nodes involved in multi-GPU setup.

    literal_predicates : list[str] or None, default None
        Predicates that identify literal (numeric) edges. When set, edges
        with these predicates are handled according to ``literal_strategy``.
        Predicate strings must match the values in the data exactly.

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
    learning_rate: float = Field(default=0.025, gt=0)
    # PyTorch backend (skip-gram, walk-sequence path) — convergence / optimiser.
    # pair_batch_size is the meaningful convergence knob: one SGD step per ~this
    # many skip-gram pairs (small → no hot predicate token is over-aggregated;
    # large → GPU-efficient). lr decays linearly learning_rate → min_learning_rate
    # over the run (classic word2vec / gensim's alpha schedule).
    optimizer: Literal["sgd", "sparseadam"] = "sgd"
    sample: float = Field(default=0.001, ge=0)          # frequent-token subsampling (gensim's `sample`); 0 disables. Default 1e-3 = gensim's classic recipe (overnight-sweep §7.4 sub_f075 validation).
    negative_sampling: Literal["uniform", "unigram"] = "unigram"   # "unigram" = classic f^0.75 negatives (overnight-sweep §7.4 sub_f075 validation; "uniform" is opt-in).
    min_learning_rate: float = Field(default=0.0001, ge=0)
    lr_decay: Literal["linear", "none"] = "linear"
    pair_batch_size: int = Field(default=32768, gt=0)
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    # torch.set_float32_matmul_precision value. "high" enables TF32 on
    # A100/A6000 Tensor Cores for ~15-25% throughput on the embedding
    # inner-products. Quality unchanged for skip-gram (200-dim dot products
    # are well within TF32 precision; sparse-scatter SGD updates aren't
    # affected). "highest" = fp32 (slower); "medium" = bf16 (faster, slight
    # quality risk). Spec 2026-05-15 D6.
    grad_clip_value: float = Field(default=0.1, ge=0)   # gradient clamp magnitude (see grad_clip_mode); 0 disables.
    grad_clip_mode: Literal["element", "row"] = "element"   # "element": |grad|≤v per component; "row": ‖grad_row‖₂≤v.
    normalize_embeddings: bool = False
    # L2-normalize each emitted embedding row at transform time. Off by default
    # (back-compat). When on: every non-zero row is rescaled to unit L2 norm;
    # zero rows (un-walked tokens per walked_mask, or gensim min_count-dropped
    # tokens) stay zero. Mathematically identical to run_geval.py's
    # `--l2-normalize` flag. RESULTS §7.1 / §7.3 show this fixes the
    # regression-Cities norm-pathology artifact (Δ ~−48% → ~−6.5% with raw
    # coverage; ~−6.9% combined with walked-only export). Adopt on export
    # when serving the trained model to scale-sensitive downstreams.
    # Thin pass-through to lightning.Trainer(accumulate_grad_batches=...). Note:
    # *increasing* this enlarges the effective batch (accumulate_grad_batches *
    # pair_batch_size pairs/step), which is generally counter-productive for
    # skip-gram — the small effective batch is what makes it converge stably.
    accumulate_grad_batches: int = Field(default=1, ge=1)
    backend: Literal["pytorch", "gensim", "torchrec"] = "pytorch"
    # Checkpointing — only consumed by backend="torchrec". Ignored by the
    # "pytorch" and "gensim" backends.
    checkpoint_dir: str | None = Field(default=None)
    checkpoint_interval_minutes: int = Field(default=30, ge=1)
    checkpoint_overwrite: bool = Field(default=False)
    checkpoint_keep_last: int = Field(default=3, ge=1)
    # library settings
    random_state: int = Field(default=42, ge=0)
    reproducible: bool = False
    multi_gpu: bool = False
    generate_artifact: bool = False
    cpu_count: int = Field(default=4, gt=0)
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
