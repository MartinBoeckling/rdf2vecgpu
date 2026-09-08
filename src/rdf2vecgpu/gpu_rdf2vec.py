import os
from typing import Optional
from cugraph import MultiGraph
import cugraph
import lightning as L
import torch
import dask
from lightning.pytorch.tuner import Tuner
from .helper.functions import _generate_vocab, cudf_to_torch_tensor
from .embedders.word2vec import SkipGram, CBOW
from .embedders.word2vec_loader import (
    SkipGramDataModule, CBOWDataModule, ParquetSkipGramDataModule, WalkSequenceDataModule,
    walks_per_pair_batch, token_counts_from_walk_parquet, token_counts_cache_path,
    load_token_counts_cache, save_token_counts_cache,
)
from .reader.kg_file_reader import KGFileReader
from .corpus.walk_corpus import SingleGPUWalkCorpus, MultiGPUWalkCorpus
from .logger import build_tracker
import cudf
from loguru import logger
from .config import RDF2VecConfig


def _token_counts_from_walk_parquet(walks_parquet: str, vocab_size: int, workers: int = 1):
    """Count token occurrences over a walk-sequences parquet dir → numpy int64
    array of length ``vocab_size`` (token id → count). Used to build the
    subsampling keep-probabilities and the f^0.75 unigram negative distribution
    for the PyTorch skip-gram path, and the unigram sampler of the torchrec path.
    Thin alias of :func:`embedders.word2vec_loader.token_counts_from_walk_parquet`
    (kept here for the existing call sites); ``workers > 1`` parallelises over
    shards. ``-1`` (cuGraph walk sentinels) are skipped, the ``0`` predicate
    padding on short walks is left in (a tiny over-count of token 0, immaterial
    for sampling weights)."""
    return token_counts_from_walk_parquet(walks_parquet, vocab_size, workers=workers)


def _token_counts_from_cudf_walks(cudf_df, vocab_size: int):
    """Token occurrence counts over an in-memory walks-as-sequences cuDF frame
    → numpy int64 array of length ``vocab_size``. Flattens the ``vertices`` /
    ``predicates`` list columns and bincounts on CPU (via numpy), dropping the
    ``-1`` cuGraph walk sentinels. Using numpy avoids CuPy JIT compilation at
    call time; the cuDF list-flatten is still GPU-side so the transfer is
    minimal (flattened 1-D integers)."""
    import numpy as np

    counts = np.zeros(vocab_size, dtype=np.int64)
    for col in ("vertices", "predicates"):
        leaves_series = cudf_df[col].list.leaves
        # Transfer to host; coerce to int64 (predicates may be float64 from
        # cuGraph's walk output — -1.0 sentinels and integer token ids).
        flat = leaves_series.to_numpy(na_value=-1)
        if np.issubdtype(flat.dtype, np.floating):
            flat = flat[~np.isnan(flat)]
        flat = flat.astype(np.int64, copy=False)
        flat = flat[flat >= 0]
        if flat.size:
            counts += np.bincount(flat, minlength=vocab_size)[:vocab_size]
    return counts


class GPU_RDF2Vec:
    def __init__(
        self,
        config: Optional[RDF2VecConfig] = None,
        client: Optional[dask.distributed.Client] = None,
        **kwargs,
    ):
        """GPU‑accelerated implementation of the RDF2Vec pipeline.

        This class wraps every step that is necessary to obtain dense vector
        representations for entities in a (potentially very large) knowledge
        graph on the GPU:

        1. **Load the triples** into a cuGraph `Graph`, optionally persisting
        intermediate artefacts.
        2. **Generate random walks** (or, in the future, BFS walks) that serve as a
        textual corpus.
        3. **Train a Word2Vec model** (currently Skip‑gram, and CBOW — note CBOW uses a full softmax, best for small/medium graphs; prefer Skip-gram or the gensim backend at large scale) on
        this corpus with all heavy‐lifting—negative sampling, matrix ops, CUDA
        kernels—performed on the GPU and orchestrated through PyTorch Lightning.
        4. **Export the learned embeddings** back to cuDF for further downstream
        analytics or as Parquet artefacts.


        Args:
            walk_strategy (str): {"random", "bfs"}
                How to sample walks from the graph.
            walk_depth (int):
                Maximum length of every walk.
            walk_number (int):
                Number of walks to start **per start vertex** (Relevant for random walk)
            embedding_model (str): {"skipgram", "cbow"}
                Variant of Word2Vec to train
            epochs (int):
                Training epochs for the Word2Vec model
            batch_size (int):
                Mini-batch size used by the Pytorch DataLoader
            vector_size (int):
                Dimensionality of the output embeddings
            window_size (int):
                Context window size in tokens
            min_count (int):
                Ignore tokens that appear fewer than this number of times when building the vocabulary
            negative_samples (int):
                Number of negative samples in the negative-sampling loss
            learning_rate (float):
                Learning rate for the optimiser
            random_state (int):
                Seed for deterministic sampling operations
            reproducible (bool):
                Turn on all Pytorch/ CUDA deterministic flags **at the cost of speed**
            multi_gpu (bool):
                If true, Dask CUDA cluster for multi-gpu training
            generate_artifact (bool):
                Persist word2idx and embedding matrices as Parquet artefacts under provided directory
            cpu_count (int):
                Number of cpu workers that feed the GPU via the DataLoader
            client (dask.distributed.Client, optional):
                Dask distributed client for multi-GPU operations. Required if multi_gpu=True.
                If None and multi_gpu=False, operates in single-GPU mode.
            tune_batch_size (bool):
                Whether to use PyTorch Lightning's Tuner to find the optimal batch size.

            number_nodes (int):
                Number of nodes in the Dask cluster for multi-GPU training.

        Attributes
        ----------
        knowledge_graph : cugraph.Graph
            Directed graph that stores the integer‑encoded triples.
        word2vec_model : torch.nn.Module or None
            Trained model after `fit`; ``None`` before.
        word2idx : cudf.DataFrame or None
            Two‑column mapping *token* → *word*; available after
            `load_data`.
        generate_artifact : bool
            Copied from the constructor.
        cpu_count : int
            Copied from the constructor.

        Raises
        ------
        NotImplementedError
            If a not‑yet‑supported walk strategy, embedding model is specified.
        ValueError
            If an unsupported file format is passed to `load_data` or if
            `transform` is called prior to `fit`.

        Examples
        --------
        >>> rdf2vec = GPU_RDF2Vec(
        ...     walk_strategy="random",
        ...     walk_depth=4,
        ...     walk_number=10,
        ...     embedding_model="skipgram",
        ...     epochs=5,
        ...     batch_size=2**14,
        ...     vector_size=256,
        ...     window_size=5,
        ...     min_count=1,
        ...     negative_samples=5,
        ...     learning_rate=0.025,
        ...     random_state=42,
        ...     reproducible=True,
        ...     multi_gpu=False,
        ...     generate_artifact=False,
        ...     cpu_count=4,
        ... )
        >>> edges = rdf2vec.load_data("example.parquet")
        >>> rdf2vec.fit(edges)
        >>> emb_df = rdf2vec.transform()
        >>> emb_df.head()

        """
        if config is None:
            config = RDF2VecConfig(**kwargs)
        # Initialize class variables
        self.config = config
        # Handle client.
        # backend='torchrec' is multi-process via torchrun (not Dask), so the
        # Dask-client requirement does not apply — torchrec ranks coordinate
        # through torch.distributed, and the pre-flight check in
        # _fit_via_torchrec enforces the torchrun launch contract.
        #
        # backend='pytorch' with multi_gpu=True uses Lightning DDP (not Dask)
        # when called via fit_from_walks(); the Dask client is only required
        # when calling fit(edge_df) (walk generation uses cuGraph + Dask-CUDA).
        # So the client is NOT enforced at construction for the pytorch backend,
        # allowing a pytorch-multi-GPU instance for fit_from_walks without a Dask
        # cluster. If you instead call fit(edge_df) on such an instance without a
        # client, the missing-cluster error surfaces later from construct_graph
        # (cuGraph multi-GPU comms not initialized), not here.
        _needs_dask_client = (
            self.config.multi_gpu
            and self.config.backend not in ("torchrec", "pytorch")
        )
        if _needs_dask_client:
            if client is None:
                raise ValueError(
                    "multi_gpu=True requires a Dask client. Please create a "
                    "LocalCUDACluster and Client, then pass the client to GPU_RDF2Vec.\n"
                    "Example:\n"
                    "  from dask_cuda import LocalCUDACluster\n"
                    "  from dask.distributed import Client\n"
                    "  cluster = LocalCUDACluster(...)\n"
                    "  client = Client(cluster)\n"
                    "  rdf2vec = GPU_RDF2Vec(..., client=client)"
                )
            self.client = client
            logger.info(
                f"Using provided Dask client with {len(client.scheduler_info()['workers'])} workers"
            )
            dask.config.set({"dataframe.backend": "cudf"})
            # Apply cuGraph 25.6 + dask 2025.5 compatibility patches
            from ._compat import apply_patches
            apply_patches()
        elif self.config.multi_gpu and self.config.backend == "pytorch" and client is not None:
            # pytorch multi-GPU with an explicit Dask client — accept it so the
            # fit(edge_df) walk-generation path (cuGraph + Dask-CUDA) still works.
            self.client = client
            logger.info(
                f"Using provided Dask client with {len(client.scheduler_info()['workers'])} workers"
            )
            dask.config.set({"dataframe.backend": "cudf"})
            from ._compat import apply_patches
            apply_patches()
        else:
            self.client = None
        self._validate_environment()
        # Initialize the cugraph graph
        self.knowledge_graph = MultiGraph(directed=True)
        self._walked_mask = None  # set during fit; consumed by transform (spec §7.5)
        self._walk_source_vertices = None  # set during fit (construct_graph); spec 2026-05-15 §1
        self.word2vec_model = None
        self.word2idx = None
        self.tracker = build_tracker(
            spec=self.config.tracker,
            kwargs=self.config.tracker_kwargs,
        )

    def read_data(
        self,
        path: str,
        col_map: Optional[dict] = None,
        read_kwargs: Optional[dict] = None,
    ):
        self.tracker.start_pipeline(
            run_name=self.config.tracker_run_name,
            tags={"lib": "gpu-rdf2vec", "backend": self.config.backend},
        )
        kg_reader = KGFileReader(
            file_path=path,
            multi_gpu=self.config.multi_gpu,
            col_map=col_map,
            read_kwargs=read_kwargs,
            walk_weighted=self.config.walk_weighted,
        )
        with self.tracker.stage("data_loading"):
            kg_data = kg_reader.read()
            self.tracker.log_params(
                {
                    "data_path": path,
                    "number_rows": kg_data.shape[0],
                    "number_columns": kg_data.shape[1],
                }
            )
        return kg_data

    def _handle_literals(self, kg_data):
        """Filter or bin literal edges identified by predicate matching.

        When ``literal_predicates`` is configured, edges whose predicate
        appears in the list are either removed (``drop``) or their object
        values are discretized into bin tokens (``bin``).
        """
        if not self.config.literal_predicates:
            return kg_data

        literal_preds = self.config.literal_predicates
        is_dask = hasattr(kg_data, "map_partitions")

        with self.tracker.stage("Literal_Handling"):
            if self.config.literal_strategy == "drop":
                mask = ~kg_data["predicate"].isin(literal_preds)
                kg_data = kg_data[mask]
                logger.info(
                    f"Dropped literal edges for predicates: {literal_preds}"
                )
                self.tracker.log_params({
                    "literal_strategy": "drop",
                    "literal_predicates": literal_preds,
                })
            elif self.config.literal_strategy == "bin":
                kg_data = self._bin_literals(kg_data, literal_preds, is_dask)

        return kg_data

    def _bin_literals(self, kg_data, literal_preds, is_dask):
        """Discretize literal object values into bin tokens."""
        literal_mask = kg_data["predicate"].isin(literal_preds)
        if is_dask:
            literal_edges = kg_data[literal_mask].compute()
            non_literal_edges = kg_data[~literal_mask]
        else:
            literal_edges = kg_data[literal_mask].copy()
            non_literal_edges = kg_data[~literal_mask]

        if len(literal_edges) == 0:
            return kg_data

        binned_parts = []
        for pred in literal_preds:
            subset = literal_edges[literal_edges["predicate"] == pred]
            if len(subset) == 0:
                continue

            numeric_vals = cudf.to_numeric(subset["object"], errors="coerce")
            valid_mask = numeric_vals.notna()
            valid_subset = subset[valid_mask].copy()
            numeric_vals = numeric_vals[valid_mask]

            if len(valid_subset) == 0:
                continue

            import cupy as cp
            n_unique = len(numeric_vals.unique())
            n_bins = min(self.config.literal_n_bins, n_unique)

            if self.config.literal_bin_strategy == "uniform":
                vmin = float(numeric_vals.min())
                vmax = float(numeric_vals.max())
                if vmin == vmax:
                    bin_labels = cudf.Series(
                        cp.zeros(len(numeric_vals), dtype="int32")
                    )
                else:
                    edges = cp.linspace(vmin, vmax, n_bins + 1)
                    bin_labels = cudf.Series(
                        cp.searchsorted(edges[1:-1], numeric_vals.to_cupy())
                    ).astype("int32")
            else:  # quantile
                quantiles = [i / n_bins for i in range(n_bins + 1)]
                edges = numeric_vals.quantile(quantiles).values_host.tolist()
                edges = sorted(set(edges))
                if len(edges) < 2:
                    bin_labels = cudf.Series(
                        cp.zeros(len(numeric_vals), dtype="int32")
                    )
                else:
                    edges_cp = cp.array(edges)
                    bin_labels = cudf.Series(
                        cp.searchsorted(edges_cp[1:-1], numeric_vals.to_cupy())
                    ).astype("int32")
            bin_labels.index = valid_subset.index

            pred_short = pred.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            valid_subset["object"] = (
                pred_short + "_bin_" + bin_labels.astype("str")
            )
            binned_parts.append(valid_subset)

        if binned_parts:
            binned_df = cudf.concat(binned_parts, ignore_index=True)
        else:
            binned_df = cudf.DataFrame(columns=non_literal_edges.columns)

        if is_dask:
            import dask_cudf
            binned_ddf = dask_cudf.from_cudf(binned_df, npartitions=1)
            result = dask_cudf.concat([non_literal_edges, binned_ddf])
        else:
            result = cudf.concat(
                [non_literal_edges, binned_df], ignore_index=True
            )

        logger.info(
            f"Binned {len(binned_df)} literal edges across "
            f"{len(literal_preds)} predicates into "
            f"{self.config.literal_n_bins} bins"
        )
        self.tracker.log_params({
            "literal_strategy": "bin",
            "literal_predicates": literal_preds,
            "literal_n_bins": self.config.literal_n_bins,
            "literal_bin_strategy": self.config.literal_bin_strategy,
            "literal_edges_binned": len(binned_df),
        })
        return result

    def generate_vocab(self, kg_data):
        with self.tracker.stage("Word2idx_Generation"):
            kg_data, word2idx = _generate_vocab(kg_data, self.config.multi_gpu)
            # Materialize dask word2idx for downstream PyTorch model creation
            if hasattr(word2idx, "compute"):
                word2idx = word2idx.compute().reset_index(drop=True)
            self.tracker.log_params({"vocab_size": word2idx.shape[0]})
        self.word2idx = word2idx
        return kg_data, word2idx

    def construct_graph(self, kg_data):
        with self.tracker.stage("Graph_Construction"):
            # cuGraph does not allow edge_attr and weight simultaneously.
            # Predicates are always looked up via merge in the walk corpus,
            # so we only need them stored in the graph for unweighted walks.
            if self.config.walk_weighted:
                edge_kwargs = dict(
                    source="subject",
                    destination="object",
                    weight="weights",
                    renumber=False,
                )
            else:
                edge_kwargs = dict(
                    source="subject",
                    destination="object",
                    edge_attr="predicate",
                    renumber=False,
                )
            if self.config.multi_gpu:
                cols = ["subject", "predicate", "object"]
                if self.config.walk_weighted:
                    cols.append("weights")
                kg_data = kg_data[cols]
                if cugraph.dask.comms.comms.is_initialized():
                    self.knowledge_graph.from_dask_cudf_edgelist(
                        kg_data, **edge_kwargs
                    )
                else:
                    logger.error(
                        "The communicator for multi-gpu cuGraph is not initalized."
                    )
                    raise ValueError(
                        "The communicator for multi-gpu cuGraph is not initalized."
                    )
            else:
                self.knowledge_graph.from_cudf_edgelist(
                    kg_data, **edge_kwargs
                )
            # Stash the actual source-vertex set for walk-gen. cuGraph's
            # `.nodes()` on this MG / renumber=False config empirically returns
            # only a fraction of the source set (~25% on the DBpedia subset).
            # Compute from the encoded edge df directly. Spec 2026-05-15 §1, D2.
            #
            # cuGraph MG API contract (verified by reading
            # cugraph/dask/sampling/uniform_random_walks.py:108-132): expects
            # start_vertices as a *single* cudf.Series (NOT dask_cudf). cuGraph
            # internally calls dask_cudf.from_cudf(start_vertices, npartitions=N)
            # to distribute across workers, then submits one walk task per
            # worker via persist_dask_df_equal_parts_per_worker. Passing a
            # dask_cudf.Series in confuses that internal from_cudf call and
            # only ~1/n_workers of starts emit walks (SLURM 246120 + 246160
            # diagnostic: walked_mask 45% on both .unique() and .drop_duplicates()).
            #
            # Materialise to a single cudf.Series. walk_corpus.py is also
            # patched to pass it through directly (not wrap to dask_cudf).
            self._walk_source_vertices = kg_data["subject"].unique()
            if hasattr(self._walk_source_vertices, "compute"):
                self._walk_source_vertices = self._walk_source_vertices.compute()
            logger.info(
                f"walk-source vertices stashed: type={type(self._walk_source_vertices).__name__} "
                f"shape={self._walk_source_vertices.shape} (cudf.Series — cuGraph distributes internally)"
            )
            if self.config.tracker != "none":
                degree_df = self.knowledge_graph.degree()
                degree_col = degree_df["degree"]
                number_nodes = self.knowledge_graph.number_of_vertices()
                number_edges = self.knowledge_graph.number_of_edges()

                self.tracker.log_metrics(
                    {
                        "number_nodes": int(number_nodes),
                        "number_edges": int(number_edges),
                        "average_degree": float(degree_col.mean()),
                        "min_degree": int(degree_col.min()),
                        "max_degree": int(degree_col.max()),
                    }
                )
                logger.debug(f"Graph has {number_edges} edges")
                logger.debug(f"Graph has {number_nodes} vertices")

    def load_data(
        self,
        path: str,
        col_map: Optional[dict] = None,
        read_kwargs: Optional[dict] = None,
    ):
        """
        Load a triple file, build the token vocabulary, and populate the internal
        cuGraph instance.

        The method chooses the most efficient cuDF reader based on the file
        extension (`.parquet`, `.csv`, `.txt`, `.nt`).  If the extension is not
        one of these, it attempts to infer any other RDF serialisation via
        `rdflib.util.guess_format` and falls back to a generic
        ``rdflib`` reader.  After reading, the triples are integer‑encoded, a
        ``word2idx`` mapping is created (and optionally persisted), and the
        resulting edge list is loaded into `self.knowledge_graph`.

        Parameters
        ----------
        path : str
            Path to the knowledge‑graph file.  Supported formats:

            * ``.parquet`` – three‑column Parquet file *(subject, predicate, object)*
            * ``.csv`` – comma‑separated triples without header
            * ``.txt`` – whitespace‑separated triples without header
            * ``.nt`` – N‑Triples (parsed with cuDF CSV reader and cleaned)
            * any other RDF serialisation recognised by
                `rdflib.util.guess_format`

        Returns
        -------
        cudf.DataFrame
            Three‑column cuDF DataFrame whose ``subject``, ``predicate``, and
            ``object`` are *int32* tokens.  The DataFrame is also stored as a
            directed edge list in `self.knowledge_graph`.

        Raises
        ------
        ValueError
            If the file extension is unsupported **and** `rdflib` cannot guess
            the RDF format.

        Notes
        -----
        * Builds a vocabulary with `_generate_vocab` and stores it in
            `self.word2idx`.
        * Persists ``word2idx`` as Parquet under *./vector/* when
            ``self.generate_artifact`` is ``True``.
        * Returns the int‑encoded edge list for downstream use (e.g.,
            a`fit`).

        """
        kg_data = self.read_data(
            path=path,
            col_map=col_map,
            read_kwargs=read_kwargs,
        )
        kg_data = self._handle_literals(kg_data)
        encoded_kg_data, word2idx = self.generate_vocab(kg_data)
        self.construct_graph(encoded_kg_data)
        return encoded_kg_data

    def _resolve_walk_vertices(self, walk_vertices):
        """Resolve the walk start-vertex set, routing around cuGraph .nodes().

        cuGraph's MG `.nodes()` on a `renumber=False` graph empirically returns
        only a fraction of the source-vertex set (~25% on the DBpedia subset);
        instead we stash the actual `kg_data["subject"].unique()` during
        `construct_graph` and consume it here. Callers can still override by
        passing `walk_vertices` explicitly. Spec 2026-05-15 §1, §2.

        Parameters
        ----------
        walk_vertices : cudf.Series, dcudf.Series, or None
            If None, returns ``self._walk_source_vertices``. If provided,
            returned unchanged (caller-supplied override).

        Returns
        -------
        cudf.Series or dcudf.Series
            The start-vertex set — int token IDs, one row per (vertex, walk)
            pair *after* the caller's downstream `.repeat(walk_number)`.

        Raises
        ------
        RuntimeError
            If `walk_vertices` is None *and* the instance is un-fitted
            (`self._walk_source_vertices is None`).
        """
        if walk_vertices is not None:
            return walk_vertices
        if self._walk_source_vertices is None:
            raise RuntimeError(
                "call fit() first, or pass walk_vertices explicitly — "
                "_walk_source_vertices is unset (instance not yet fitted)"
            )
        return self._walk_source_vertices

    def walk_generation(
        self, edge_df: cudf.DataFrame, walk_vertices: cudf.Series = None
    ) -> cudf.DataFrame:
        with self.tracker.stage("Walk_Generation"):
            if self.config.multi_gpu:
                walk_instance = MultiGPUWalkCorpus(
                    self.knowledge_graph,
                    self.config.window_size,
                    walk_weighted=self.config.walk_weighted,
                )
            else:
                walk_instance = SingleGPUWalkCorpus(
                    self.knowledge_graph,
                    self.config.window_size,
                    walk_weighted=self.config.walk_weighted,
                )
            if self.config.walk_strategy == "random":
                walk_vertices = self._resolve_walk_vertices(walk_vertices)
                # Repeat each vertex walk_number times for walk start points
                n = self.config.walk_number
                if hasattr(walk_vertices, "map_partitions"):
                    walk_vertices = walk_vertices.map_partitions(
                        lambda s: s.repeat(n).reset_index(drop=True)
                    )
                else:
                    walk_vertices = walk_vertices.repeat(n)
                walk_corpus = walk_instance.random_walk(
                    edge_df=edge_df,
                    walk_vertices=walk_vertices,
                    walk_depth=self.config.walk_depth,
                    random_state=self.config.random_state,
                    word2vec_model=self.config.embedding_model,
                    min_count=self.config.min_count,
                )
            elif self.config.walk_strategy == "bfs":
                walk_vertices = self._resolve_walk_vertices(walk_vertices)
                walk_corpus = walk_instance.bfs_walk(
                    edge_df=edge_df,
                    walk_vertices=walk_vertices,
                    walk_depth=self.config.walk_depth,
                    random_state=self.config.random_state,
                    word2vec_model=self.config.embedding_model,
                    min_count=self.config.min_count,
                )
            self.tracker.log_params(
                {
                    "walk_depth": self.config.walk_depth,
                    "random_state": self.config.random_state,
                    "walk_strategy": self.config.walk_strategy,
                    "walk_number": self.config.walk_number,
                    "walk_weighted": self.config.walk_weighted,
                    "min_count": self.config.min_count,
                }
            )

        return walk_corpus

    def walk_generation_sequences(
        self, edge_df: cudf.DataFrame, walk_vertices: cudf.Series = None
    ):
        """Walks-as-sequences output for the gensim training backend.

        Sister to `walk_generation`, but emits walks as parallel
        `vertices: list<int>` + `predicates: list<int>` columns (one row per
        walk) instead of pre-built (center, context) skip-gram pairs.
        Gensim builds its own skip-gram windows internally, so it wants the
        raw walk sequences, not pre-computed pairs.

        Only the "random" walk strategy is supported for the gensim backend
        in this release — bfs walks would need a parallel BFS-to-sequences
        path on the corpus classes.
        """
        with self.tracker.stage("Walk_Generation"):
            if self.config.multi_gpu:
                walk_instance = MultiGPUWalkCorpus(
                    self.knowledge_graph,
                    self.config.window_size,
                    walk_weighted=self.config.walk_weighted,
                )
            else:
                walk_instance = SingleGPUWalkCorpus(
                    self.knowledge_graph,
                    self.config.window_size,
                    walk_weighted=self.config.walk_weighted,
                )
            if self.config.walk_strategy != "random":
                raise NotImplementedError(
                    f"backend='gensim' currently supports walk_strategy='random' "
                    f"only (got {self.config.walk_strategy!r}). bfs+gensim would "
                    f"need a parallel sequences output on bfs_walk."
                )
            walk_vertices = self._resolve_walk_vertices(walk_vertices)
            n = self.config.walk_number
            if hasattr(walk_vertices, "map_partitions"):
                walk_vertices = walk_vertices.map_partitions(
                    lambda s: s.repeat(n).reset_index(drop=True)
                )
            else:
                walk_vertices = walk_vertices.repeat(n)
            walk_sequences = walk_instance.random_walk_sequences(
                walk_vertices=walk_vertices,
                walk_depth=self.config.walk_depth,
                random_state=self.config.random_state,
            )
            self.tracker.log_params(
                {
                    "walk_depth": self.config.walk_depth,
                    "random_state": self.config.random_state,
                    "walk_strategy": self.config.walk_strategy,
                    "walk_number": self.config.walk_number,
                    "walk_weighted": self.config.walk_weighted,
                    "min_count": self.config.min_count,
                    "backend": "gensim",
                }
            )
        return walk_sequences

    def fit(self, edge_df: cudf.DataFrame, walk_vertices: cudf.Series = None) -> None:
        """
         Train a Word2Vec model on random-walk sequences generated from the
         knowledge graph.

         The method performs three high-level steps:

         1. **Generate walks** – Uses the configured ``walk_strategy``
         to obtain (center, context) pairs that mimic natural-language contexts.
         2. **Build the training data set** – Converts the pairs to PyTorch
         tensors and wraps them in a performant `TensorDataset`/`DataLoader`.
         3. **Optimize the embedding model** – Instantiates the requested
         Word2Vec variant, then fits it with a
         PyTorch Lightning `Trainer`.

         Parameters
         ----------
         edge_df : cudf.DataFrame
             Int-encoded edge list with columns ``subject``, ``predicate``,
             ``object``—typically the output of `load_data`.
         walk_vertices : cudf.Series or None, default None
             Optional subset of starting vertices from which to launch random
             walks. If None, all vertices in ``self.knowledge_graph`` are used.

         Raises
         ------
         ValueError
             If an invalid ``walk_strategy`` is supplied.

        Notes
        -----
        - Walks are repeated ``self.walk_number`` times and have maximum depth ``self.walk_depth``.
        - Skip-gram training uses negative sampling with ``self.negative_samples`` negatives per positive pair.
        - The trained model is stored in ``self.word2vec_model`` and can subsequently be exported via ``transform``.

         Examples
         --------
         >>> edges = rdf2vec.load_data("example.parquet")
         >>> rdf2vec.fit(edges)
        """
        # backend dispatch: gensim consumes walks-as-sequences via parquet
        # streaming and trains on CPU; pytorch consumes (center, context)
        # skip-gram pairs and trains on GPU via Lightning.
        if self.config.backend == "torchrec":
            self._fit_via_torchrec(edge_df, walk_vertices)
            return
        if self.config.backend == "gensim":
            walk_sequences = self.walk_generation_sequences(
                edge_df=edge_df, walk_vertices=walk_vertices
            )
            self._fit_via_gensim(walk_sequences)
            return
        # [spike] pytorch backend, skip-gram, random walks → the new walk-sequence
        # path (sparse embeddings + SGD + on-GPU pair expansion). CBOW / bfs fall
        # through to the legacy pre-expand-to-pairs path below.
        if (
            self.config.backend == "pytorch"
            and self.config.embedding_model == "skipgram"
            and self.config.walk_strategy == "random"
        ):
            walk_sequences = self.walk_generation_sequences(
                edge_df=edge_df, walk_vertices=walk_vertices
            )
            self._fit_via_pytorch_sequences(walk_sequences)
            return
        walk_corpus = self.walk_generation(
            edge_df=edge_df,
            walk_vertices=walk_vertices,
        )
        # In multi-GPU mode, checkpoint the distributed walk corpus to
        # parquet and tear down Dask before training.  This avoids
        # materializing the entire corpus to a single GPU (cudf row limit)
        # and frees GPU memory held by the Dask cluster for training.
        if self.config.multi_gpu and hasattr(walk_corpus, "to_parquet"):
            import tempfile, shutil
            from dask.distributed import wait as dask_wait
            # Persist to flatten the dask-expr graph, then write to parquet.
            # Each Dask worker writes its partition — no single GPU holds all data.
            walk_corpus = walk_corpus.persist()
            dask_wait(walk_corpus)
            self._walk_dir = tempfile.mkdtemp(prefix="rdf2vec_walks_")
            walk_path = f"{self._walk_dir}/corpus"
            logger.info(f"Checkpointing walk corpus to {walk_path}")
            walk_corpus.to_parquet(walk_path, write_index=False)
            del walk_corpus
            # Tear down Dask comms + cluster to free GPU memory for DDP training
            try:
                import cugraph.dask.comms.comms as Comms
                if Comms.is_initialized():
                    Comms.destroy()
            except Exception:
                pass
            if self.client is not None:
                self.client.close()
                self.client = None
            logger.info("Dask cluster shut down, starting DDP training")
            self._fit_from_parquet(walk_path)
            shutil.rmtree(self._walk_dir, ignore_errors=True)
        else:
            # Single-GPU path: materialize and train in-process
            if hasattr(walk_corpus, "compute"):
                walk_corpus = walk_corpus.compute()
            self._fit_from_tensors(walk_corpus)

    def _fit_via_gensim(self, walk_sequences) -> None:
        """Train Word2Vec via the optional gensim backend.

        Materializes the walks-as-sequences corpus to a temp parquet directory
        (gensim's streaming `WalkCorpus` iterator reads parquet files), tears
        down the dask cluster if multi-GPU was used so the GPU memory is
        released before CPU-side training, then fits gensim's
        negative-sampling Word2Vec.

        The trained gensim model is stored on `self.word2vec_model` (same
        attribute the pytorch backend uses); `transform()` branches on
        backend to extract embeddings into the standard `[token, word,
        embedding_*]` cudf DataFrame.
        """
        from .embedders.gensim_word2vec import train_gensim_word2vec
        import tempfile, shutil

        with self.tracker.stage("Word2Vec_Training"):
            walk_dir = tempfile.mkdtemp(prefix="rdf2vec_gensim_walks_")
            try:
                walks_parquet = f"{walk_dir}/walks"
                logger.info(f"Checkpointing walk sequences to {walks_parquet}")
                if hasattr(walk_sequences, "persist"):
                    # dask_cudf.DataFrame (multi_gpu): each worker writes its
                    # partition. Discriminate on `.persist` — NOT `.to_parquet`,
                    # which a single-GPU cudf.DataFrame also has, so the old
                    # check sent cudf into this dask-only branch and crashed on
                    # `.persist()`.
                    # `random_walk_sequences()` already persist+wait-ed the
                    # walk output, but re-pin defensively here so a future
                    # refactor that moves the persist out of the corpus class
                    # doesn't silently re-roll the stochastic walk generator
                    # on this `to_parquet`. Matches the explicit persist+wait
                    # the PyTorch checkpoint path uses.
                    from dask.distributed import wait as dask_wait

                    walk_sequences = walk_sequences.persist()
                    dask_wait(walk_sequences)
                    walk_sequences.to_parquet(walks_parquet, write_index=False)
                else:
                    # cudf.DataFrame: single-file write under the directory.
                    import os as _os
                    _os.makedirs(walks_parquet, exist_ok=True)
                    walk_sequences.to_parquet(
                        f"{walks_parquet}/part.0.parquet", index=False
                    )

                # Build walked_mask before tearing down dask / training: identical
                # logic to the pytorch path, used here for the transform-time sanity
                # assertion. See spec Components §3.
                logger.info("Counting walk-corpus token frequencies for coverage mask...")
                _vocab = self.word2idx.compute().shape[0] if hasattr(self.word2idx, "compute") else self.word2idx.shape[0]
                _counts = _token_counts_from_walk_parquet(walks_parquet, _vocab)
                self._walked_mask = (_counts > 0)
                logger.info(f"  walked_mask: {int(self._walked_mask.sum()):,}/{self._walked_mask.shape[0]:,} tokens touched by walks")

                # Free GPU resources before CPU-side gensim training. Mirrors
                # the multi-GPU pytorch path's teardown so the gensim workers
                # have full host RAM.
                if self.config.multi_gpu:
                    try:
                        import cugraph.dask.comms.comms as Comms

                        if Comms.is_initialized():
                            Comms.destroy()
                    except Exception:
                        pass
                    if self.client is not None:
                        self.client.close()
                        self.client = None
                    logger.info(
                        "Dask cluster shut down; starting gensim training"
                    )

                logger.info(
                    f"Training gensim Word2Vec "
                    f"(vector_size={self.config.vector_size}, "
                    f"window={self.config.window_size}, "
                    f"epochs={self.config.epochs}, "
                    f"workers={self.config.cpu_count})"
                )
                gensim_model = train_gensim_word2vec(
                    walks_dir=walks_parquet,
                    word2idx_df=self.word2idx,
                    vector_size=self.config.vector_size,
                    window=self.config.window_size,
                    min_count=self.config.min_count,
                    embedding_model=self.config.embedding_model,
                    negative=self.config.negative_samples,
                    epochs=self.config.epochs,
                    workers=self.config.cpu_count,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                )
                self.word2vec_model = gensim_model
                self.tracker.log_params(
                    {
                        "vector_size": self.config.vector_size,
                        "window_size": self.config.window_size,
                        "epochs": self.config.epochs,
                        "negative_samples": self.config.negative_samples,
                        "embedding_model": self.config.embedding_model,
                        "learning_rate": self.config.learning_rate,
                        "backend": "gensim",
                    }
                )
            finally:
                shutil.rmtree(walk_dir, ignore_errors=True)

    def _load_corpus_vocab(self, walks_path: str, vocab_path: str, *, load_frame: bool = True):
        """Load ``(vocab_df, token_counts, vocab_path)`` from a prepared corpus.
        With ``load_frame=False`` ``vocab_df`` is ``None`` and only the row
        count is read (from the parquet footer).

        The decoupled torchrec flow: the vocabulary is READ from the
        on-disk parquet produced by :meth:`prepare_corpus` — it is NOT
        rebuilt from the edge list here (that needs cuGraph / dask-cudf,
        which cannot run under the torchrun job; see
        :meth:`prepare_corpus`). Token-frequency counts are derived from
        the walks parquet for the f^0.75 unigram negative sampler.

        Parameters
        ----------
        walks_path : str
            Parquet directory of walks-as-sequences (used only to derive
            token counts).
        vocab_path : str
            Parquet file with columns ``token_id`` (int64) and ``entity``
            (string), as written by :meth:`prepare_corpus`.

        Returns
        -------
        vocab_df : cudf.DataFrame
            Columns ``token_id`` (int64) and ``entity`` (string), one row
            per token — the layout
            ``torchrec_word2vec.extract_embeddings_to_dask_cudf`` joins on.
        token_counts : torch.Tensor
            ``int64`` tensor of shape ``(vocab_size,)``.
        vocab_path : str
            Echoed back unchanged; it is the DCP resume identity key (see
            ``torchrec_word2vec._maybe_resume_checkpoint``).
        """
        if load_frame:
            vocab_df = cudf.read_parquet(vocab_path)
            vocab_size = vocab_df.shape[0]
        else:
            # Only the row count is needed (torchrec training never touches the
            # entity strings — 393M IRIs, ~20 GB of GPU memory per rank at
            # Wikidata scale); it comes from the parquet footer, no data read.
            import pyarrow.parquet as pq
            vocab_df = None
            vocab_size = int(pq.read_metadata(vocab_path).num_rows)

        # Token-frequency counts over the walk parquet → torch.LongTensor.
        # Prefer the on-disk cache next to the vocab: counting means streaming
        # the whole walks corpus on CPU, and under torchrun EVERY rank would do
        # it before a single training step (10 h per rank at Wikidata scale).
        # Precompute once with performance/helix/make_token_counts.py.
        cache = token_counts_cache_path(vocab_path)
        counts_np = load_token_counts_cache(cache, vocab_size)
        if counts_np is None:
            logger.warning(
                f"token-count cache {cache} missing or stale — counting over "
                f"{walks_path} in-process (every rank repeats this; precompute it "
                f"once with performance/helix/make_token_counts.py)"
            )
            counts_np = _token_counts_from_walk_parquet(
                walks_path, vocab_size, workers=min(self.config.cpu_count, 8)
            )
            if os.environ.get("RANK", "0") == "0":
                save_token_counts_cache(cache, counts_np)
        else:
            logger.info(f"token counts: loaded cache {cache} ({vocab_size:,} tokens)")
        token_counts = torch.from_numpy(counts_np).to(torch.long)

        return vocab_df, token_counts, vocab_path

    def _load_word2idx(self, vocab_path: str):
        """Load a prepared corpus vocab (`token_id`, `entity` — as written by
        `prepare_corpus`) into `self.word2idx` in the `(token, word)` layout the
        pytorch / gensim extract paths expect, and return it. The walks were
        encoded against these same token ids, so the mapping is consistent by
        construction."""
        vocab = cudf.read_parquet(vocab_path)
        self.word2idx = vocab.rename(columns={"token_id": "token", "entity": "word"})[
            ["token", "word"]
        ]
        return self.word2idx

    def _build_walk_pair_loader(self, walks_path: str, vocab_df):
        """DataLoader yielding ``(center, context)`` long-tensor batches.

        Wraps the existing :class:`WalkSequenceDataModule` (which yields
        ``(vertices, predicates)`` walk-row batches in int32) with a thin
        on-GPU pair-expansion stage that materialises the skip-gram pairs
        each step. The expansion mirrors
        :meth:`SkipGram._expand_skipgram` (in ``embedders/word2vec.py``),
        minus the frequent-token subsampling — torchrec's fused optimizer
        handles row-wise scaling that obviates the simpler subsampling
        path, and the f^0.75 unigram negatives in
        :func:`_sample_negatives` already de-bias hot tokens.

        The pair-template builder now lives in ``embedders/word2vec_loader``
        (:class:`_PairsFromWalks`); ``SkipGram._expand_skipgram`` still has
        its own copy of the windowing template.
        """
        from torch.utils.data import DataLoader
        from .embedders.word2vec_loader import walks_per_pair_batch

        cfg = self.config
        walks_per_batch = walks_per_pair_batch(
            cfg.pair_batch_size, cfg.walk_depth, cfg.window_size
        )

        # WalkSequenceDataModule shards parquet files per-DDP-rank via
        # its `setup()` hook, but outside a Lightning Trainer (the
        # torchrec case — Fabric, not Trainer) `setup()` leaves all
        # files visible to all ranks. Skip the DataModule entirely and
        # use the underlying iterable directly; emulate the rank-shard
        # logic via torch.distributed below.
        from .embedders.word2vec_loader import (
            _PairsFromWalks,
            _ThreadPrefetch,
            _WalkSeqIterable,
        )
        # pair_batch_size: cut walk batches by their actual skip-gram pair
        # count, not by a row count derived from full-length walks. Real walks
        # are short (Wikidata: 17 of the nominal 60 pairs per walk, dbpedia2016:
        # 19), so the row-count batches carried under a third of the configured
        # pairs and cost 3.5x the steps. emit_batch_size is the fallback when
        # pair_batch_size is unset.
        ds = _WalkSeqIterable(
            walks_path,
            emit_batch_size=walks_per_batch,
            read_chunk_size=262_144,
            shuffle=True,
            seed=cfg.random_state,
            pair_batch_size=cfg.pair_batch_size,
            window=cfg.window_size,
        )
        # Rank-shard at the ROW level, not the file level. torchrec's
        # sharded forward issues an all_to_all on EVERY step, so every rank
        # MUST execute the same number of forward() calls or that collective
        # desyncs and the job deadlocks. File-level sharding violates that
        # invariant two ways: with fewer files than ranks a rank gets zero
        # data (and previously raised, killing its peers mid-collective),
        # and with uneven file sizes ranks get different batch counts.
        # Instead every rank reads ALL files — the inner walk-batch stream,
        # and therefore the emitted batch COUNT, is identical across ranks
        # because the shuffle seed is rank-independent — and each rank keeps
        # a disjoint row stride of every batch (see _PairsFromWalks). Equal
        # batch counts by construction; each rank still trains on
        # ~1/world_size of the pairs. rank/world_size are read here in the
        # main process: DataLoader workers don't join the process group, so
        # dist.is_initialized() is False inside them.
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Pair expansion (module-level _PairsFromWalks: rank-strides each walk
        # batch, expands to (center, context) int64 on the CPU), then a
        # thread prefetch that runs decode + expansion a few batches ahead of
        # the GPU step. num_workers stays 0: this loader runs inside the
        # torchrec training process, which has CUDA + NCCL initialised, and a
        # forked DataLoader worker touching CUDA aborts with
        # cudaErrorInitializationError — a thread has no such problem and the
        # heavy parts (pyarrow decode, numpy copies) release the GIL. Iteration
        # stays deterministic and identical across ranks, so every rank yields
        # the same batch COUNT (the training loop min-reduces a continue flag
        # as a backstop).
        pair_ds = _ThreadPrefetch(
            _PairsFromWalks(ds, window=cfg.window_size, rank=rank, world_size=world_size),
            depth=4,
        )
        return DataLoader(
            pair_ds,
            batch_size=None,         # iterable yields already-batched pairs
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

    def _torchrec_preflight(self) -> None:
        """Validate config + environment for the torchrec training backend.

        Shared by :meth:`fit_from_walks`. Raises early so the user sees
        configuration errors immediately rather than mid-run. See
        ``docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md``
        §5 for the full error-handling contract.
        """
        # Pre-flight: model coverage
        if self.config.embedding_model != "skipgram":
            raise NotImplementedError(
                f"backend='torchrec' currently supports embedding_model='skipgram' "
                f"only (got {self.config.embedding_model!r})."
            )

        # Pre-flight: multi-GPU is mandatory
        if not self.config.multi_gpu:
            raise ValueError(
                "backend='torchrec' requires multi_gpu=True "
                "(the sharded embedding path is multi-process by construction)."
            )

        # Pre-flight: reject SGD. The fused optimizer-in-backward applies the
        # update INSIDE the FBGEMM kernel, where gradient clipping is a no-op —
        # the `gradient_clipping`/`max_gradient` kwargs are silently dropped (they
        # belong to a separate GradientClippingOptimizer wrapper that only clips a
        # standalone `.step()`, not a fused-in-backward update). So torchrec SGD
        # runs UNCLIPPED, and the SUM skip-gram loss makes hot-token gradients
        # overshoot → divergence to garbage on non-trivial vocabularies (verified
        # on wikidata5m: per-batch loss 33k → 1.79M; a 100×-tighter clip had zero
        # effect, confirming the clip is dead). grad_clip_value does NOT make it
        # safe — that earlier guard rested on a false premise. PartialRowWiseAdam
        # is per-row adaptive and stable without clipping. Refuse silent garbage.
        if self.config.optimizer == "sgd":
            raise ValueError(
                "backend='torchrec' does not support optimizer='sgd': the fused "
                "optimizer-in-backward cannot apply gradient clipping, so the SUM "
                "skip-gram loss makes unclipped SGD diverge on non-trivial "
                "vocabularies. Use optimizer='rowwiseadagrad' (per-row Adagrad — "
                "gradient-following, stable, recovers structure given enough steps) "
                "or 'sparseadam' (PartialRowWiseAdam)."
            )

        # Pre-flight: imports
        try:
            import torchrec  # noqa: F401
            import fbgemm_gpu  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='torchrec' requires the torchrec extra. "
                "Install with: pip install rdf2vecgpu[torchrec]"
            ) from e

        # Pre-flight: distributed init
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError(
                "backend='torchrec' requires a distributed launch. "
                "Run via: torchrun --standalone --nproc_per_node=N your_script.py"
            )

    def _fit_via_torchrec(
        self, edge_df, walk_vertices: cudf.Series | None = None
    ) -> None:
        """``fit(edge_df)`` is unsupported for backend='torchrec'.

        Data preparation (walk generation + vocab build) relies on a
        Dask-CUDA cluster / cuGraph dask comms that cannot coexist with
        the torchrun/NCCL process group torchrec training runs under —
        each multi-GPU paradigm wants to own all GPUs. Prepare the corpus
        once (outside torchrun) via :meth:`prepare_corpus`, then train via
        :meth:`fit_from_walks`.
        """
        raise RuntimeError(
            "backend='torchrec' does not support fit(edge_df) — walk generation "
            "and vocab build cannot run inside the torchrun training job "
            "(Dask-CUDA / cuGraph and torchrun/NCCL cannot share a process). "
            "Prepare the corpus separately, then train:\n"
            "  # 1. prepare (normal process; single- or multi-GPU):\n"
            "  edges = model.load_data(path)\n"
            "  walks_path, vocab_path = model.prepare_corpus(edges, out_dir)\n"
            "  # 2. train (under torchrun --nproc_per_node=N):\n"
            "  model.fit_from_walks(walks_path, vocab_path)"
        )

    def fit_from_walks(self, walks_path: str, vocab_path: str) -> None:
        """Train a Word2Vec model from a prepared on-disk walk corpus.

        The decoupled training entry point: prepare a corpus once via
        :meth:`prepare_corpus` (or any walk-generation step that writes
        the standard ``(vertices: list<int>, predicates: list<int>)``
        parquet layout), then call this method to train any supported
        backend from it.

        Supported backends (``config.backend``):

        * **gensim** — streams walk sequences from the parquet directory
          and trains gensim's negative-sampling Word2Vec on CPU.
          ``gensim`` is an OPTIONAL extra and is imported lazily here so
          that installations without it are unaffected.
        * **pytorch (single-GPU)** — reads the corpus into a GPU-resident
          cuDF frame and trains via :class:`InMemoryWalkSeqDataModule` +
          the skip-gram SGD path.
        * **pytorch (multi-GPU)** — streams the parquet per-rank under DDP
          using :class:`WalkSequenceDataModule`.
        * **torchrec** — runs the row-wise sharded torchrec skip-gram
          model under ``torchrun``. MUST be launched via
          ``torchrun --standalone --nproc_per_node=N``.

        Does NOT generate walks or build vocab — see :meth:`prepare_corpus`
        for why those are a separate step (cuGraph / Dask-CUDA and
        torchrun/NCCL cannot share a process). After training, call
        :meth:`transform` to extract embeddings.

        Parameters
        ----------
        walks_path : str
            Parquet directory of walks-as-sequences, as returned by
            :meth:`prepare_corpus`. Must resolve to the same data on every
            rank (a shared filesystem for multi-node).
        vocab_path : str
            Parquet file mapping ``token_id`` ↔ ``entity``, as returned by
            :meth:`prepare_corpus`. Also the DCP resume identity key for
            the torchrec backend.

        Raises
        ------
        FileNotFoundError
            If ``walks_path`` or ``vocab_path`` does not exist on disk.
        ValueError
            If an unsupported backend is specified.
        RuntimeError
            (torchrec only) If not launched under ``torchrun``.
        """
        import os
        # Backend config preflight (torchrec: cbow unsupported / must run under
        # torchrun) fires BEFORE any disk read, so a misconfiguration raises its
        # specific error instead of being masked by a FileNotFoundError when the
        # paths also happen to be missing.
        backend = self.config.backend
        if backend == "torchrec":
            self._torchrec_preflight()

        if not os.path.exists(walks_path):
            raise FileNotFoundError(f"walks_path does not exist: {walks_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab_path does not exist: {vocab_path}")

        if backend == "torchrec":
            from rdf2vecgpu.embedders.torchrec_word2vec import train_torchrec_word2vec
            train_torchrec_word2vec(self, walks_path, vocab_path)
            return

        self._load_word2idx(vocab_path)

        if backend == "gensim":
            from .embedders.gensim_word2vec import train_gensim_word2vec   # lazy: optional extra
            # Coverage mask for transform() (zeros un-walked rows) — parity with
            # the fit() gensim path (_fit_via_gensim), which sets _walked_mask
            # from the walk-corpus token counts.
            self._walked_mask = (
                _token_counts_from_walk_parquet(walks_path, self.word2idx.shape[0]) > 0
            )
            with self.tracker.stage("Word2Vec_Training"):
                self.word2vec_model = train_gensim_word2vec(
                    walks_dir=walks_path, word2idx_df=self.word2idx,
                    vector_size=self.config.vector_size, window=self.config.window_size,
                    min_count=self.config.min_count, embedding_model=self.config.embedding_model,
                    negative=self.config.negative_samples, epochs=self.config.epochs,
                    workers=self.config.cpu_count, learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                )
            return

        if backend == "pytorch":
            from .embedders.word2vec_loader import (
                InMemoryWalkSeqDataModule, WalkSequenceDataModule, walks_per_pair_batch,
            )
            walks_per_batch = walks_per_pair_batch(
                self.config.pair_batch_size, self.config.walk_depth, self.config.window_size
            )
            if self.config.multi_gpu:
                import glob as _glob
                import pyarrow.parquet as _pq
                n_gpus = torch.cuda.device_count()
                token_counts = _token_counts_from_walk_parquet(walks_path, self.word2idx.shape[0])
                files = _glob.glob(f"{walks_path}/*.parquet")
                n_walks = sum(_pq.read_metadata(f).num_rows for f in files) or 1
                # WalkSequenceDataModule row-strides the corpus across DDP ranks
                # and caps each rank to n_walks // n_gpus rows (regardless of how
                # many parquet files the corpus has), so each rank runs
                # ceil((n_walks // n_gpus) / walks_per_batch) steps. Size the LR
                # schedule for that exact per-rank step count.
                per_rank_walks = n_walks // n_gpus
                steps_per_epoch = max(1, -(-per_rank_walks // walks_per_batch))
                dm = WalkSequenceDataModule(
                    walks_path, batch_size=walks_per_batch,
                    num_workers=self.config.cpu_count, shuffle=True, seed=self.config.random_state,
                )
            else:
                df = cudf.read_parquet(walks_path)
                n_gpus = 1
                token_counts = _token_counts_from_cudf_walks(df, self.word2idx.shape[0])
                steps_per_epoch = max(1, -(-int(len(df)) // walks_per_batch))
                dm = InMemoryWalkSeqDataModule(
                    df, emit_batch_size=walks_per_batch,
                    shuffle=True, seed=self.config.random_state,
                )
            self._walked_mask = (token_counts > 0)
            with self.tracker.stage("Word2Vec_Training"):
                self._train_pytorch_sequences(
                    dm, token_counts=token_counts,
                    steps_per_epoch=steps_per_epoch, n_gpus=n_gpus,
                )
            return

        raise ValueError(f"fit_from_walks: unsupported backend {backend!r}")

    def prepare_corpus(
        self, edge_df, out_dir: str, walk_vertices: cudf.Series = None
    ) -> tuple[str, str]:
        """Generate a walk corpus + vocab and persist them under ``out_dir``.

        The data-preparation half of the decoupled torchrec flow. Run this
        OUTSIDE ``torchrun`` — in a normal process, single-GPU
        (``multi_gpu=False``) or on a Dask-CUDA cluster
        (``multi_gpu=True``). The torchrec training job then consumes the
        result via :meth:`fit_from_walks`. Walk generation and vocab build
        rely on cuGraph / dask-cudf, which cannot run inside the
        torchrun/NCCL process group torchrec trains under — hence the split.

        Requires ``load_data`` (or an equivalent) to have already populated
        ``self.word2idx`` and built ``self.knowledge_graph``; the usual
        call sequence is::

            edges = model.load_data(path)
            walks_path, vocab_path = model.prepare_corpus(edges, out_dir)

        Parameters
        ----------
        edge_df
            The int-encoded edge frame returned by ``load_data``.
        out_dir : str
            Directory to write ``walks/`` (parquet dir) and
            ``vocab.parquet`` into. Created if absent.
        walk_vertices : cudf.Series, optional
            Override for the walk start-vertex set; defaults to the stashed
            subject set (see :meth:`_resolve_walk_vertices`).

        Returns
        -------
        (walks_path, vocab_path) : tuple[str, str]
            Absolute paths to the walks parquet directory and the vocab
            parquet file — pass these to :meth:`fit_from_walks`.
        """
        import os
        from .embedders.torchrec_word2vec import (
            _materialize_walks_sequences_to_parquet,
        )

        os.makedirs(out_dir, exist_ok=True)

        # Walks-as-sequences → parquet (stable, addressable path).
        walks_seq = self.walk_generation_sequences(
            edge_df=edge_df, walk_vertices=walk_vertices,
        )
        walks_path = _materialize_walks_sequences_to_parquet(
            walks_seq, walks_parquet=os.path.join(out_dir, "walks"),
        )

        # Vocab → (token_id: int64, entity: string) parquet. word2idx was
        # built upstream by load_data() → generate_vocab() and may still be
        # a dask_cudf in multi-GPU prep mode; materialise before projecting.
        word2idx = self.word2idx
        if hasattr(word2idx, "compute"):
            word2idx = word2idx.compute()
        vocab_df = cudf.DataFrame({
            "token_id": word2idx["token"].astype("int64"),
            "entity":   word2idx["word"],
        })
        vocab_path = os.path.join(out_dir, "vocab.parquet")
        vocab_df.to_parquet(vocab_path, index=False)

        return walks_path, vocab_path

    def _fit_via_pytorch_sequences(self, walk_sequences) -> None:
        """Train PyTorch skip-gram from walks-as-sequences: checkpoint the
        sequences to a temp parquet dir, tear down dask if multi-GPU, then train
        single-GPU via WalkSequenceDataModule + SkipGram. The DataModule yields
        small walk-row batches sized to ~``config.pair_batch_size`` skip-gram
        pairs each (so no hot predicate token is over-aggregated), and SkipGram
        uses SGD + a linear LR decay (``config.learning_rate`` →
        ``config.min_learning_rate``) stepped per optimizer step — classic
        word2vec / gensim's alpha schedule."""
        import tempfile, shutil, os
        from dask.distributed import wait as dask_wait

        # TF32 on Tensor Cores for the embedding inner-products (spec 2026-05-15 D6).
        # "high" is the recommended Lightning setting; quality unchanged for skip-gram.
        torch.set_float32_matmul_precision(self.config.matmul_precision)

        from .embedders.word2vec_loader import InMemoryWalkSeqDataModule, walks_per_pair_batch

        if not self.config.multi_gpu:
            # ── Single-GPU: GPU-resident in-memory feed (no parquet, no CPU) ──
            if hasattr(walk_sequences, "compute"):
                walk_sequences = walk_sequences.compute()
            if hasattr(self.word2idx, "compute"):
                self.word2idx = self.word2idx.compute()
            walks_per_batch = walks_per_pair_batch(
                self.config.pair_batch_size, self.config.walk_depth, self.config.window_size
            )
            n_walks = int(len(walk_sequences))
            steps_per_epoch = max(1, -(-n_walks // walks_per_batch))
            token_counts = _token_counts_from_cudf_walks(walk_sequences, self.word2idx.shape[0])
            self._walked_mask = (token_counts > 0)
            dm = InMemoryWalkSeqDataModule(
                walk_sequences, emit_batch_size=walks_per_batch,
                shuffle=True, seed=self.config.random_state,
            )
            with self.tracker.stage("Word2Vec_Training"):
                self._train_pytorch_sequences(
                    dm, token_counts=token_counts,
                    steps_per_epoch=steps_per_epoch, n_gpus=1,
                )
            return

        # Single-GPU training even when the (now-torn-down) dask cluster used
        # several GPUs for walk-gen: Lightning's SLURM DDP launcher needs
        # --ntasks-per-node == devices, which doesn't hold for a 1-task SLURM
        # job with --gres=gpu:N; and the SkipGram (sparse Embedding, ~vocab×dim)
        # fits one 48 GB GPU. Multi-GPU DDP for this path is a follow-up.
        n_gpus = 1
        with self.tracker.stage("Word2Vec_Training"):
            walk_dir = tempfile.mkdtemp(prefix="rdf2vec_pt_walks_")
            try:
                walks_parquet = f"{walk_dir}/walks"
                logger.info(f"Checkpointing walk sequences to {walks_parquet}")
                # discriminate dask_cudf vs cudf via `.persist` — both have `to_parquet`.
                if hasattr(walk_sequences, "persist"):
                    walk_sequences = walk_sequences.persist()
                    dask_wait(walk_sequences)
                    n_walks = int(len(walk_sequences))
                    walk_sequences.to_parquet(walks_parquet, write_index=False)
                else:
                    n_walks = int(len(walk_sequences))
                    os.makedirs(walks_parquet, exist_ok=True)
                    walk_sequences.to_parquet(f"{walks_parquet}/part.0.parquet", index=False)
                if self.config.multi_gpu:
                    try:
                        import cugraph.dask.comms.comms as Comms
                        if Comms.is_initialized():
                            Comms.destroy()
                    except Exception:
                        pass
                    if self.client is not None:
                        self.client.close()
                        self.client = None
                    logger.info("Dask cluster shut down; starting PyTorch training")
                # word2idx may be dask_cudf in multi-GPU mode → materialise for vocab size
                if hasattr(self.word2idx, "compute"):
                    self.word2idx = self.word2idx.compute()
                if self.config.reproducible:
                    L.seed_everything(self.config.random_state, workers=True)

                # Size walk-row batches so each ≈ pair_batch_size skip-gram pairs.
                walks_per_batch = walks_per_pair_batch(
                    self.config.pair_batch_size, self.config.walk_depth, self.config.window_size
                )
                steps_per_epoch = max(1, -(-n_walks // walks_per_batch))   # ceil

                # Token-frequency table — always built. Two consumers:
                # (1) subsampling / f^0.75 negatives, when enabled;
                # (2) walked_mask for the coverage-report fix (spec §7.5) —
                #     transform zeros un-walked rows so rdf2vecgpu_pytorch's
                #     coverage report matches rdf2vecgpu_gensim's behaviour.
                # Cost: one CPU pass over the walk parquet (~15 s on the full
                # DBpedia subset; ~3 h training elsewhere → negligible).
                logger.info("Counting walk-corpus token frequencies (for subsampling / unigram negatives / coverage mask)...")
                token_counts = _token_counts_from_walk_parquet(walks_parquet, self.word2idx.shape[0])
                logger.info(f"  token counts: total={int(token_counts.sum()):,} | "
                            f"nonzero={int((token_counts > 0).sum()):,}/{token_counts.shape[0]:,} | "
                            f"max={int(token_counts.max()):,}")
                # walked_mask: True iff this token appeared as a vertex or predicate
                # in at least one walk. Stored for transform() to zero un-walked rows.
                self._walked_mask = (token_counts > 0)
                logger.info(f"  walked_mask: {int(self._walked_mask.sum()):,}/{self._walked_mask.shape[0]:,} tokens touched by walks")

                logger.info(
                    f"PyTorch skip-gram: {n_walks:,} walks | walks/batch={walks_per_batch} "
                    f"(~{self.config.pair_batch_size} pairs/step) | steps/epoch≈{steps_per_epoch} | "
                    f"epochs={self.config.epochs} | optimizer={self.config.optimizer} | "
                    f"lr {self.config.learning_rate}→{self.config.min_learning_rate} ({self.config.lr_decay}) | "
                    f"accumulate_grad_batches={self.config.accumulate_grad_batches} | "
                    f"grad_clip={self.config.grad_clip_value}/{self.config.grad_clip_mode} | "
                    f"sample={self.config.sample} | negative_sampling={self.config.negative_sampling}"
                )

                datamodule = WalkSequenceDataModule(
                    walks_parquet,
                    batch_size=walks_per_batch,
                    num_workers=self.config.cpu_count,
                    shuffle=True,
                    seed=self.config.random_state,
                )
                self._train_pytorch_sequences(
                    datamodule, token_counts=token_counts,
                    steps_per_epoch=steps_per_epoch, n_gpus=n_gpus,
                )
            finally:
                shutil.rmtree(walk_dir, ignore_errors=True)

    def _train_pytorch_sequences(self, datamodule, *, token_counts, steps_per_epoch, n_gpus):
        """Build the SkipGram model + Lightning Trainer and fit on `datamodule`.
        Shared by `_fit_via_pytorch_sequences` (fit()) and `fit_from_walks`.
        `datamodule` yields `(vertices, predicates)` batches."""
        # Sparse embeddings are incompatible with NCCL all_reduce (DDP multi-GPU):
        # NCCL requires dense tensors for gradient synchronization. Use dense
        # embeddings when training with n_gpus > 1; sparse for single-GPU to
        # preserve memory efficiency (only touched rows allocated in gradients).
        use_sparse = n_gpus == 1
        model = SkipGram(
            vocab_size=self.word2idx.shape[0],
            embedding_dim=self.config.vector_size,
            neg_samples=self.config.negative_samples,
            learning_rate=self.config.learning_rate,
            use_sparse=use_sparse,
            epochs=self.config.epochs,
            min_learning_rate=self.config.min_learning_rate,
            lr_decay=self.config.lr_decay,
            window=self.config.window_size,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            steps_per_epoch=steps_per_epoch,
            grad_clip_value=self.config.grad_clip_value,
            optimizer=self.config.optimizer,
            grad_clip_mode=self.config.grad_clip_mode,
            sample=self.config.sample,
            negative_sampling=self.config.negative_sampling,
            token_counts=token_counts,
        )
        self.tracker.log_pytorch()
        # Multi-GPU DDP: force Lightning to self-spawn one worker per GPU, even
        # inside a SLURM allocation. By default Lightning auto-detects SLURM
        # (SLURM_NTASKS) and defers process launch to SLURM; under `--ntasks=1`
        # that silently collapses DDP to a single rank ("MEMBER: 1/1"). Pinning a
        # LightningEnvironment (creates_processes_externally=False) restores the
        # plain-`python` subprocess-launch path this backend is designed for.
        if n_gpus > 1:
            from lightning.pytorch.strategies import DDPStrategy
            from lightning.pytorch.plugins.environments import LightningEnvironment
            strategy = DDPStrategy(cluster_environment=LightningEnvironment())
        else:
            strategy = "auto"
        trainer = L.Trainer(
            max_epochs=self.config.epochs,
            log_every_n_steps=200,
            accelerator="gpu",
            precision=32,
            devices=n_gpus,
            strategy=strategy,
            num_nodes=self.config.num_nodes,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, datamodule)
        self.tracker.log_model_pytorch(model, artifact_path="word2vec_model")
        self.word2vec_model = model
        self.tracker.log_params({
            "vector_size": self.config.vector_size,
            "window_size": self.config.window_size,
            "epochs": self.config.epochs,
            "negative_samples": self.config.negative_samples,
            "embedding_model": "skipgram",
            "optimizer": self.config.optimizer,
            "learning_rate": self.config.learning_rate,
            "min_learning_rate": self.config.min_learning_rate,
            "lr_decay": self.config.lr_decay,
            "pair_batch_size": self.config.pair_batch_size,
            "accumulate_grad_batches": self.config.accumulate_grad_batches,
            "grad_clip_value": self.config.grad_clip_value,
            "grad_clip_mode": self.config.grad_clip_mode,
            "sample": self.config.sample,
            "negative_sampling": self.config.negative_sampling,
            "backend": "pytorch",
            "corpus": "walk_sequences_onfly_gpu",
        })

    def _fit_from_parquet(self, walk_path: str) -> None:
        """Train Word2Vec from parquet shards using DDP across all GPUs."""
        n_gpus = torch.cuda.device_count()
        with self.tracker.stage("Word2Vec_Training"):
            if self.config.embedding_model == "skipgram":
                word2vec_model = SkipGram(
                    vocab_size=self.word2idx.shape[0],
                    embedding_dim=self.config.vector_size,
                    neg_samples=self.config.negative_samples,
                    learning_rate=self.config.learning_rate,
                )
                datamodule = ParquetSkipGramDataModule(
                    parquet_path=walk_path,
                    batch_size=self.config.batch_size or 512,
                )
            elif self.config.embedding_model == "cbow":
                # CBOW multi-GPU parquet path: read on single GPU and use
                # the existing CBOW DataModule (CBOW list aggregation not
                # yet supported from parquet shards)
                walk_corpus = cudf.read_parquet(walk_path)
                self._fit_from_tensors(walk_corpus)
                return
            else:
                raise ValueError(
                    f"Unsupported embedding model: {self.config.embedding_model}."
                )
            if self.config.reproducible:
                L.seed_everything(self.config.random_state, workers=True)
            self.tracker.log_pytorch()
            trainer = L.Trainer(
                max_epochs=self.config.epochs,
                log_every_n_steps=1,
                accelerator="gpu",
                precision="16-mixed",
                devices=n_gpus,
                strategy="ddp" if n_gpus > 1 else "auto",
                num_nodes=self.config.num_nodes,
                enable_checkpointing=False,
            )
            trainer.fit(word2vec_model, datamodule)
            self.tracker.log_model_pytorch(
                word2vec_model, artifact_path="word2vec_model"
            )
            self.word2vec_model = word2vec_model

    def _fit_from_tensors(self, walk_corpus) -> None:
        """Train Word2Vec from an in-memory cudf DataFrame (single-GPU)."""
        center_tensor = cudf_to_torch_tensor(walk_corpus, "center")
        with self.tracker.stage("Word2Vec_Training"):
            if self.config.embedding_model == "skipgram":
                context_tensor = cudf_to_torch_tensor(walk_corpus, "context")
                word2vec_model = SkipGram(
                    vocab_size=self.word2idx.shape[0],
                    embedding_dim=self.config.vector_size,
                    neg_samples=self.config.negative_samples,
                    learning_rate=self.config.learning_rate,
                )
                datamodule = SkipGramDataModule(
                    center_tensor=center_tensor,
                    context_tensor=context_tensor,
                    batch_size=(
                        self.config.batch_size
                        if self.config.batch_size
                        else max(1, len(context_tensor) // 100)
                    ),
                )
            elif self.config.embedding_model == "cbow":
                word2vec_model = CBOW(
                    vocab_size=self.word2idx.shape[0],
                    embedding_dim=self.config.vector_size,
                    learning_rate=self.config.learning_rate,
                )
                context_series = walk_corpus["context"]
                exploded_df = (
                    context_series.to_frame("context").explode("context").reset_index()
                )
                exploded_df["pos"] = exploded_df.groupby("index").cumcount()
                pivot_df = exploded_df.pivot(
                    index="index", columns="pos", values="context"
                )
                reset_pivot_df = pivot_df.reset_index(drop=True)
                reset_pivot_df = reset_pivot_df.fillna(-1).astype("int32")
                context_tensor = torch.utils.dlpack.from_dlpack(
                    reset_pivot_df.to_dlpack()
                ).contiguous()
                datamodule = CBOWDataModule(
                    center_tensor=center_tensor,
                    context_tensor=context_tensor,
                    batch_size=(
                        self.config.batch_size
                        if self.config.batch_size
                        else max(1, len(context_tensor) // 100)
                    ),
                )
            else:
                raise ValueError(
                    f"Unsupported embedding model: {self.config.embedding_model}."
                )
            if self.config.reproducible:
                L.seed_everything(self.config.random_state, workers=True)
            self.tracker.log_pytorch()
            trainer = L.Trainer(
                max_epochs=self.config.epochs,
                log_every_n_steps=1,
                accelerator="gpu",
                precision="16-mixed",
                devices=1,
                num_nodes=self.config.num_nodes,
                enable_checkpointing=False,
            )
            if self.config.tune_batch_size:
                tuner = Tuner(trainer)
                tuner.scale_batch_size(
                    word2vec_model,
                    mode="power",
                    datamodule=datamodule,
                    steps_per_trial=1,
                    init_val=max(1, len(center_tensor) // 200),
                    max_trials=12,
                )
            trainer.fit(word2vec_model, datamodule)
            self.tracker.log_model_pytorch(
                word2vec_model, artifact_path="word2vec_model"
            )
            self.word2vec_model = word2vec_model

    def transform(self) -> cudf.DataFrame:
        """
        Convert the learned Word2Vec parameters into a cuDF table of
        entity‑level embeddings.

        The method fetches the trained embedding matrix
        (shape ``[vocab_size, vector_size]``), converts it from the PyTorch
        tensor on the GPU into a cuDF DataFrame, and concatenates it with the
        ``word2idx`` mapping so every row contains

        * ``token`` – integer ID
        * ``word`` – original IRI / literal
        * ``embedding_0 … embedding_{vector_size‑1}`` – float32 components

        When `self.generate_artifact` is *True*, the resulting table is also
        written to *./vector/embeddings_<model‑hash>.parquet*.

        Returns
        -------
        cudf.DataFrame
            A ``(vocab_size × (vector_size + 2))`` DataFrame with the mapping
            and embeddings.

        Raises
        ------
        ValueError
            If called before `fit` (no trained model) **or**
            `load_data` (no ``word2idx`` vocabulary).

        Notes
        -----
        The method is a pure transformation; it never mutates the underlying
        Word2Vec parameters.  Use it to obtain fresh snapshots after each
        training run.

        """
        if self.config.backend == "torchrec":
            if not hasattr(self, "_torchrec_model"):
                raise RuntimeError(
                    "transform() on backend='torchrec' must be called within the "
                    "same distributed launch as fit(). The sharded model is "
                    "rank-local and cannot be reconstructed after the process "
                    "group is destroyed."
                )
            from rdf2vecgpu.embedders.torchrec_word2vec import (
                extract_embeddings_to_dask_cudf,
            )
            vocab_df = self._torchrec_vocab_df
            if vocab_df is None:
                # Training loads only the vocab size; the entity strings are
                # read here, once, when they are actually joined.
                vocab_df = cudf.read_parquet(self._torchrec_vocab_path)
            return extract_embeddings_to_dask_cudf(self._torchrec_model, vocab_df)
        # Check if model is fitted and word2idx is available
        with self.tracker.stage("Embedding_Extraction"):
            if self.word2vec_model is None or self.word2idx is None:
                raise ValueError(
                    "The transform method is not possible to call without a fitted model or a generated word2idx setup. "
                    "Please call the 'fit' method first or the 'load_data' method to generate the word2idx setup."
                )
            if self.config.backend == "gensim":
                embedding_df = self._extract_embeddings_gensim()
            else:
                embedding_df = self._extract_embeddings_pytorch()
            if self.config.generate_artifact:
                embedding_df.to_parquet(
                    f"vector/embeddings_{self.config.embedding_model}.parquet",
                    index=False,
                )
            return embedding_df

    def _extract_embeddings_pytorch(self) -> cudf.DataFrame:
        """Pull `[token, word, embedding_0..n]` cudf DataFrame from the
        PyTorch trainer's in-embeddings weight matrix.

        When `self._walked_mask` is set (i.e. the model was fit on this
        instance — the normal path), rows for tokens that never appeared in
        any walk are zeroed before emission. This matches gensim's
        behaviour (gensim drops un-walked tokens via min_count → its
        transform emits zeros for them via `kv_to_token_aligned_matrix`'s
        default-zero fill) and makes `run_geval.py`'s all-zero-drop rule
        kick in downstream. See spec
        `docs/superpowers/specs/2026-05-14-rdf2vecgpu-coverage-report-fix-design.md`
        (Components §3 — "`transform` zeros un-walked rows"); RESULTS.md §7.5 is the
        corresponding empirical results section.
        """
        import cupy as cp

        model_embeddings = (
            self.word2vec_model.get_embeddings().detach().cuda()
        )
        # Convert via CuPy to avoid cudf from_dlpack column-major requirement;
        # .copy() gives us an owned buffer — cp.from_dlpack is zero-copy and
        # would otherwise share GPU memory with the live model weights.
        cp_arr = cp.from_dlpack(model_embeddings).copy()
        if self._walked_mask is not None:
            # Stay on-GPU: convert numpy bool mask to cupy, then zero in place
            # on the OWNED copy (we copied above so the live model weights are
            # not mutated — preserves transform()'s "pure transformation" contract).
            unwalked_idx = cp.asarray(~self._walked_mask)
            cp_arr[unwalked_idx] = 0.0
        if self.config.normalize_embeddings:
            # L2-normalize each row; zero rows stay zero (divide by safe_norms=1
            # for them so 0.0 / 1.0 = 0.0). Mathematically identical to
            # run_geval.py --l2-normalize. RESULTS §7.1 / §7.3.
            norms = cp.linalg.norm(cp_arr, axis=1, keepdims=True)
            safe_norms = cp.where(norms > 0, norms, 1.0)
            cp_arr = cp_arr / safe_norms
        model_embeddings_df = cudf.DataFrame(
            {str(i): cp_arr[:, i] for i in range(cp_arr.shape[1])}
        )
        model_embeddings_df.columns = model_embeddings_df.columns.astype(str)
        model_embeddings_df = model_embeddings_df.add_prefix("embedding_")
        return cudf.concat([self.word2idx, model_embeddings_df], axis=1)

    def _extract_embeddings_gensim(self) -> cudf.DataFrame:
        """Pull `[token, word, embedding_0..n]` cudf DataFrame from the
        gensim KeyedVectors output.

        Row alignment is delegated to
        `embedders.gensim_word2vec.kv_to_token_aligned_matrix`, which is a
        pure-numpy helper unit-tested separately. The cudf concat below joins
        the resulting token-ordered matrix with a token-sorted `word2idx` so
        each row is `[token, word, embedding_*]` for the same token id.

        Note: gensim's `min_count` filter (when `min_count > 1`) drops tokens
        from the KV; those rows get a zero embedding here. The PyTorch path
        keeps its random-init weight in the same situation. Both behaviors
        are technically lossy and worth normalizing in a follow-up.
        """
        import cupy as cp
        from .embedders.gensim_word2vec import kv_to_token_aligned_matrix

        kv = self.word2vec_model.wv  # gensim.models.KeyedVectors
        # Numeric, token-indexed matrix (zeros for any token gensim filtered).
        vectors = kv_to_token_aligned_matrix(kv, self.word2idx)
        if len(kv.index_to_key) < vectors.shape[0]:
            logger.warning(
                f"gensim KV vocab ({len(kv.index_to_key):,}) is smaller than "
                f"word2idx ({vectors.shape[0]:,}); "
                f"{vectors.shape[0] - len(kv.index_to_key):,} tokens were "
                f"filtered by min_count and will appear as zero embeddings."
            )
        # Sanity: every token gensim's KV emits must be in walked_mask. This
        # catches a code regression where the bitmask logic and gensim's
        # corpus-vocab logic diverge — they should be the same set (gensim's
        # min_count drops tokens with corpus count < min_count; walked_mask
        # drops tokens with corpus count == 0). See spec Components §3.
        if self._walked_mask is not None:
            import numpy as np
            emitted_keys = set(kv.index_to_key)
            # Map keys back to token ids via word2idx (token id == row index).
            w2i = self.word2idx
            if hasattr(w2i, "compute"):
                w2i = w2i.compute()
            word_col = w2i["word"].to_pandas() if hasattr(w2i["word"], "to_pandas") else w2i["word"]
            token_col = w2i["token"].to_pandas() if hasattr(w2i["token"], "to_pandas") else w2i["token"]
            word_to_token = dict(zip(word_col, token_col))
            emitted_ids = {word_to_token[k] for k in emitted_keys if k in word_to_token}
            walked_ids = set(np.flatnonzero(self._walked_mask).tolist())
            stray = emitted_ids - walked_ids
            assert not stray, (
                f"gensim emitted {len(stray)} token id(s) not in walked_mask: "
                f"{sorted(stray)[:10]}... — vocab/walk-parquet mismatch (spec Components §3)"
            )
        cp_arr = cp.asarray(vectors)
        if self.config.normalize_embeddings:
            # L2-normalize each row; zero rows (gensim min_count-dropped) stay
            # zero. Symmetric with the pytorch extract path above.
            norms = cp.linalg.norm(cp_arr, axis=1, keepdims=True)
            safe_norms = cp.where(norms > 0, norms, 1.0)
            cp_arr = cp_arr / safe_norms
        model_embeddings_df = cudf.DataFrame(
            {str(i): cp_arr[:, i] for i in range(cp_arr.shape[1])}
        )
        model_embeddings_df.columns = model_embeddings_df.columns.astype(str)
        model_embeddings_df = model_embeddings_df.add_prefix("embedding_")
        # `self.word2idx` may be dask_cudf; materialize to cudf for the concat,
        # and sort by token so the row order matches the matrix above.
        if hasattr(self.word2idx, "compute"):
            w2i = self.word2idx.compute()
        else:
            w2i = self.word2idx
        w2i = w2i.sort_values("token").reset_index(drop=True)
        return cudf.concat([w2i, model_embeddings_df], axis=1)

    def fit_transform(
        self, edge_df: cudf.DataFrame, walk_vertices: cudf.DataFrame = None
    ) -> cudf.DataFrame:
        """
        Train the Word2Vec model **and** immediately return the resulting
        embeddings.

        This convenience wrapper simply calls `fit` followed by
        `transform`.  Use it when you do **not** need to inspect the
        model object itself and only care about the final entity vectors.

        Parameters
        ----------
        edge_df : cudf.DataFrame
            Int‑encoded triples that define the knowledge graph (typically the
            output of `load_data`).
        walk_vertices : cudf.Series or None, default ``None``
            Optional subset of start vertices for walk generation; see
            a`fit` for semantics.

        Returns
        -------
        cudf.DataFrame
            The concatenated ``word2idx``–embedding table produced by
            a`transform`.

        Notes
        -----
        - All exceptions raised by a`fit` or a`transform`
          propagate unchanged.
        - The trained model is still stored in
          `self.word2vec_model` for later re‑use.

        """
        self.fit(edge_df, walk_vertices)
        embedding_df = self.transform()
        return embedding_df

    def close(self):
        """Close the Dask client, cuGraph Comms, and cluster if they exist."""
        self.tracker.close()
        if self.client is not None:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def save(self, path: str) -> None:
        """Persist config, vocabulary, and trained model to *path*.

        Creates a directory with:
          - config.json     (RDF2VecConfig as JSON)
          - word2idx.parquet (token↔word mapping)
          - model.pt        (PyTorch state dict, only if fit() was called)
        """
        from pathlib import Path as _P
        import json

        out = _P(path)
        out.mkdir(parents=True, exist_ok=True)
        # Config
        (out / "config.json").write_text(self.config.model_dump_json(indent=2))
        # Vocabulary
        if self.word2idx is not None:
            self.word2idx.to_parquet(out / "word2idx.parquet", index=False)
        # Model weights
        if self.word2vec_model is not None:
            torch.save(self.word2vec_model.state_dict(), out / "model.pt")
        logger.info(f"Saved GPU_RDF2Vec artifacts to {out}")

    @classmethod
    def load(cls, path: str, client=None) -> "GPU_RDF2Vec":
        """Restore a GPU_RDF2Vec instance from a previously saved directory."""
        from pathlib import Path as _P
        import json

        src = _P(path)
        config = RDF2VecConfig.model_validate_json((src / "config.json").read_text())
        instance = cls(config=config, client=client)
        # Vocabulary
        word2idx_path = src / "word2idx.parquet"
        if word2idx_path.exists():
            instance.word2idx = cudf.read_parquet(word2idx_path)
        # Model weights
        model_path = src / "model.pt"
        if model_path.exists():
            if config.embedding_model == "skipgram":
                model = SkipGram(
                    vocab_size=instance.word2idx.shape[0],
                    embedding_dim=config.vector_size,
                    neg_samples=config.negative_samples,
                    learning_rate=config.learning_rate,
                )
            elif config.embedding_model == "cbow":
                model = CBOW(
                    vocab_size=instance.word2idx.shape[0],
                    embedding_dim=config.vector_size,
                    learning_rate=config.learning_rate,
                )
            else:
                raise ValueError(f"Unknown embedding_model: {config.embedding_model}")
            model.load_state_dict(torch.load(model_path, weights_only=True))
            instance.word2vec_model = model
        logger.info(f"Loaded GPU_RDF2Vec artifacts from {src}")
        return instance

    def _validate_environment(self):
        """Validates the configuration parameters for the GPU_RDF2Vec class.

        This method checks the validity of various configuration parameters, ensuring
        they meet the expected types, ranges, and constraints. If any parameter is invalid,
        an appropriate exception is raised.

            EnvironmentError: If CUDA is not available on the system.

            ValueError: If `multi_gpu` is True but no Dask client is provided.

            Warning: If `multi_gpu` is True but fewer than 1 GPU is detected.
        """

        if not torch.cuda.is_available():
            raise EnvironmentError(
                "CUDA is not available. A GPU is required to run this code."
            )
        if self.config.multi_gpu and torch.cuda.device_count() < 2:
            logger.warning(
                f"multi_gpu=True but torch reports {torch.cuda.device_count()} visible GPU on this process."
            )
