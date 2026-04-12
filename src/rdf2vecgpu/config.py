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
        Learning rate used by the optimizer.

    epochs : int, default 5
        Number of training epochs.

    batch_size : int or None, default None
        Explicit batch size; if None, Lightning's tuner may pick one.

    tune_batch_size : bool, default True
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
    learning_rate: float = Field(default=0.0001, gt=0)
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
