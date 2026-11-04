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
from .embedders.word2vec_loader import SkipGramDataModule, CBOWDataModule, ParquetSkipGramDataModule
from .reader.kg_file_reader import KGFileReader
from .corpus.walk_corpus import (
    SingleGPUWalkCorpus,
    MultiGPUWalkCorpus,
    encode_predicates,
    predicate_code_table,
    weighted_edge_index,
)
from .logger import build_tracker
import cudf
from loguru import logger
from .config import RDF2VecConfig
from dataclasses import dataclass


@dataclass(frozen=True)
class WalkParquet:
    """Walk sequences already written to a parquet directory by the multi-GPU
    `walk_generation_sequences(out_dir=...)` path."""

    path: str
    n_walks: int


_NO_GRAPH = (
    "no graph built yet: call load_data() before fit(); read_data() only reads the file, "
    "load_data() also builds the vocabulary and the graph"
)


def _ddp_strategy(n_gpus: int, num_nodes: int):
    """Lightning strategy for training on `n_gpus` GPUs per node.

    On one node, Lightning spawns one worker per GPU itself. Inside a SLURM
    allocation it would otherwise defer the launch to SLURM and, under
    `--ntasks=1`, run DDP as a single rank. When SLURM starts several tasks
    (one per GPU), the launch stays with SLURM."""
    if n_gpus < 2:
        return "auto"
    if num_nodes == 1 and int(os.environ.get("SLURM_NTASKS", "1")) == 1:
        from lightning.pytorch.strategies import DDPStrategy
        from lightning.pytorch.plugins.environments import LightningEnvironment

        return DDPStrategy(cluster_environment=LightningEnvironment())
    return "ddp"


def _check_walk_files(walks: WalkParquet) -> None:
    """Raise unless the parquet files under `walks.path` hold all
    `walks.n_walks` walks. The dask workers write them, so on a cluster
    spanning several nodes the files of other nodes are not visible here."""
    from glob import glob
    import pyarrow.parquet as pq

    files = glob(os.path.join(walks.path, "**", "*.parquet"), recursive=True)
    rows = sum(pq.read_metadata(f).num_rows for f in files)
    if rows != walks.n_walks:
        raise RuntimeError(
            f"{walks.path} holds {rows:,} of the {walks.n_walks:,} walks written; "
            "multi-GPU gensim training needs every dask worker on this node"
        )


def _checked_weights(kg_data):
    """`kg_data` with float `weights`, rejecting missing, negative or
    non-finite weights, and graphs whose weights are all 0 (no weighted walk
    would leave its start vertex)."""
    w = kg_data["weights"]
    if w.dtype.kind != "f":
        kg_data = kg_data.assign(weights=w.astype("float32"))
        w = kg_data["weights"]
    n_bad = (w.isnull() | (w < 0) | ~(w.abs() < float("inf"))).sum()
    n_positive = (w > 0).sum()
    if hasattr(n_bad, "compute"):
        n_bad, n_positive = dask.compute(n_bad, n_positive)
    if int(n_bad):
        raise ValueError(f"{int(n_bad)} edge weights are missing, negative or not finite")
    if not int(n_positive):
        raise ValueError("all edge weights are 0, so weighted walks would not leave their start vertices")
    return kg_data


class GPU_RDF2Vec:
    def __init__(
        self,
        config: Optional[RDF2VecConfig] = None,
        client: Optional[dask.distributed.Client] = None,
        **kwargs,
    ):
        """GPU-accelerated implementation of the RDF2Vec pipeline.

        The class wraps every step needed to obtain vector representations
        for the entities of a (potentially very large) knowledge graph:

        1. **Load the triples** (`load_data`): read the file, build the
           token vocabulary (``word2idx``) and the cuGraph ``MultiGraph``.
        2. **Generate walks** (`fit`): random or BFS walks on the GPU with
           cuGraph, starting from every subject of the graph by default.
        3. **Train Word2Vec** (`fit`): Skip-gram or CBOW, either on the
           GPU(s) with PyTorch Lightning (``backend="pytorch"``) or on the CPU
           with gensim (``backend="gensim"``).
        4. **Export the embeddings** (`transform`): a cuDF table with one row
           per vocabulary token, optionally written to Parquet.

        Parameters
        ----------
        config : RDF2VecConfig, optional
            All hyperparameters; see `RDF2VecConfig` for the full list and
            the defaults. If omitted, one is built from ``**kwargs``.
        client : dask.distributed.Client, optional
            Dask client of a running cluster (for example a
            ``dask_cuda.LocalCUDACluster``). Required when
            ``multi_gpu=True``, ignored otherwise.
        **kwargs
            Forwarded to ``RDF2VecConfig(**kwargs)`` when ``config`` is None.

        Attributes
        ----------
        config : RDF2VecConfig
            The validated configuration.
        knowledge_graph : cugraph.MultiGraph
            Directed multigraph of the integer-encoded triples; built by
            `load_data`.
        word2idx : cudf.DataFrame or None
            Vocabulary with columns ``token`` (int id) and ``word`` (IRI or
            literal); available after `load_data`.
        word2vec_model : torch.nn.Module, gensim.models.Word2Vec or None
            The trained model after `fit`; None before.

        Raises
        ------
        EnvironmentError
            If no CUDA GPU is available.
        ValueError
            If ``multi_gpu=True`` and no ``client`` is given.
        pydantic.ValidationError
            If a keyword argument is not a valid `RDF2VecConfig` value.

        Examples
        --------
        >>> rdf2vec = GPU_RDF2Vec(
        ...     walk_strategy="random",
        ...     walk_depth=4,
        ...     walk_number=10,
        ...     embedding_model="skipgram",
        ...     epochs=5,
        ...     vector_size=128,
        ...     learning_rate=0.01,
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
        # Handle client
        if self.config.multi_gpu:
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
        else:
            self.client = None
        self._validate_environment()
        # Initialize the cugraph graph
        self.knowledge_graph = MultiGraph(directed=True)
        self._walk_source_vertices = None  # set during fit (construct_graph); spec 2026-05-15 §1
        self._code_to_token = None  # predicate code -> token table (construct_graph)
        self._edge_index = None  # weighted walks' predicate lookup table (construct_graph)
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
        """Read a triple file into a cuDF (or, with ``multi_gpu``, Dask-cuDF)
        DataFrame with ``subject``, ``predicate`` and ``object`` columns, plus
        ``weights`` when ``walk_weighted=True``.

        This only reads the file. Use `load_data` to also build the
        vocabulary and the graph, which `fit` needs.

        Parameters
        ----------
        path : str
            Path to the knowledge-graph file; see `load_data` for the
            supported formats.
        col_map : dict, optional
            Renames the file's columns to ``subject``, ``predicate``,
            ``object`` (and ``weights``), for example
            ``{"src": "subject", "rel": "predicate", "dst": "object"}``.
        read_kwargs : dict, optional
            Extra keyword arguments for the cuDF/Dask CSV, Parquet or ORC
            reader (for example ``{"delimiter": "\\t"}``). Ignored for
            N-Triples/N-Quads and rdflib formats. ``columns`` is rejected for
            Parquet; use ``col_map`` instead.

        Returns
        -------
        cudf.DataFrame or dask_cudf.DataFrame
            The triples with string values, not yet integer-encoded.
        """
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
            if self.config.multi_gpu:
                # The edge list is still the lazy read plus vocabulary merges,
                # and several steps below compute it.
                kg_data = kg_data.persist()
            # cuGraph does not allow edge_attr and weight simultaneously, so
            # weighted walks look their predicates up in `_edge_index` by
            # (subject, object, weight) instead.
            if self.config.walk_weighted:
                edge_kwargs = dict(
                    source="subject",
                    destination="object",
                    weight="weights",
                    renumber=False,
                )
            else:
                # The edge attribute travels in cuGraph's float32 weight slot,
                # exact only below 2^24, so the graph carries compact predicate
                # codes; walk output is decoded back with `_code_to_token`.
                edge_kwargs = dict(
                    source="subject",
                    destination="object",
                    edge_attr="pcode",
                    renumber=False,
                )
            if not self.config.walk_weighted:
                tokens = kg_data["predicate"].unique()
                if hasattr(tokens, "compute"):
                    tokens = tokens.compute()
                tokens = tokens.sort_values().values_host.astype("int64")
                self._code_to_token = predicate_code_table(tokens)
                if self.config.multi_gpu:
                    kg_data = kg_data.assign(pcode=kg_data["predicate"].map_partitions(
                        encode_predicates, tokens, meta=("pcode", "float32")))
                else:
                    kg_data = kg_data.assign(pcode=encode_predicates(kg_data["predicate"], tokens))
            else:
                self._code_to_token = None
                kg_data = _checked_weights(kg_data)
                self._edge_index = weighted_edge_index(kg_data)
                if self.config.multi_gpu:
                    self._edge_index = self._edge_index.persist()
            if self.config.multi_gpu:
                cols = ["subject", "object"]
                cols.append("weights" if self.config.walk_weighted else "pcode")
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
        Load a triple file, build the token vocabulary, and populate the
        internal cuGraph graph.

        Runs `read_data`, the literal handling (``literal_predicates``), the
        vocabulary construction and the graph construction. Call it before
        `fit` or `fit_transform`.

        Parameters
        ----------
        path : str
            Path to the knowledge-graph file. The reader is chosen by the file
            extension:

            * ``.parquet``, ``.orc`` – columns ``subject``, ``predicate``,
              ``object`` (rename others with ``col_map``)
            * ``.csv`` – a header row naming the columns ``subject``,
              ``predicate``, ``object`` (or names mapped with ``col_map``)
            * ``.nt``, ``.nq`` – N-Triples / N-Quads, read line by line on the
              GPU; angle brackets are stripped from IRIs, the graph term of
              N-Quads is dropped, and lines with a blank-node subject or
              predicate are skipped
            * any other RDF serialisation that `rdflib.util.guess_format`
              recognises (for example ``.ttl``, ``.rdf``, ``.jsonld``),
              parsed on the CPU with rdflib
        col_map : dict, optional
            See `read_data`.
        read_kwargs : dict, optional
            See `read_data`.

        Returns
        -------
        cudf.DataFrame or dask_cudf.DataFrame
            Edge list whose ``subject``, ``predicate`` and ``object`` are
            integer tokens (see ``self.word2idx``). Pass it to `fit`.

        Raises
        ------
        NotImplementedError
            If the file extension is not supported and rdflib cannot guess
            the RDF format.
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
            If `walk_vertices` is None *and* no graph has been built yet
            (`load_data` / `construct_graph` not called).
        """
        if walk_vertices is not None:
            return walk_vertices
        if self._walk_source_vertices is None:
            raise RuntimeError(_NO_GRAPH)
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
                    code_to_token=self._code_to_token,
                    edge_index=self._edge_index,
                )
            else:
                walk_instance = SingleGPUWalkCorpus(
                    self.knowledge_graph,
                    self.config.window_size,
                    walk_weighted=self.config.walk_weighted,
                    code_to_token=self._code_to_token,
                    edge_index=self._edge_index,
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
        self, edge_df: cudf.DataFrame, walk_vertices: cudf.Series = None, out_dir: Optional[str] = None
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

        With ``multi_gpu`` and `out_dir`, the walks are written by the GPUs
        straight to that parquet directory in chunks
        (`MultiGPUWalkCorpus.write_walk_sequences`) and a `WalkParquet` is
        returned instead of a dask frame.
        """
        with self.tracker.stage("Walk_Generation"):
            if self.config.multi_gpu:
                walk_instance = MultiGPUWalkCorpus(
                    self.knowledge_graph,
                    self.config.window_size,
                    walk_weighted=self.config.walk_weighted,
                    code_to_token=self._code_to_token,
                    edge_index=self._edge_index,
                )
            else:
                walk_instance = SingleGPUWalkCorpus(
                    self.knowledge_graph,
                    self.config.window_size,
                    walk_weighted=self.config.walk_weighted,
                    code_to_token=self._code_to_token,
                    edge_index=self._edge_index,
                )
            if self.config.walk_strategy != "random":
                raise NotImplementedError(
                    f"backend='gensim' currently supports walk_strategy='random' "
                    f"only (got {self.config.walk_strategy!r}). bfs+gensim would "
                    f"need a parallel sequences output on bfs_walk."
                )
            walk_vertices = self._resolve_walk_vertices(walk_vertices)
            n = self.config.walk_number
            if self.config.multi_gpu and out_dir is not None:
                n_walks = walk_instance.write_walk_sequences(
                    walk_vertices,
                    walk_number=n,
                    walk_depth=self.config.walk_depth,
                    random_state=self.config.random_state,
                    out_dir=out_dir,
                )
                walk_sequences = WalkParquet(out_dir, n_walks)
            else:
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
        Generate walks from the knowledge graph and train the Word2Vec model.

        With ``backend="pytorch"`` the walks are turned into (center, context)
        pairs on the GPU and trained with a PyTorch Lightning ``Trainer`` (DDP
        across all GPUs with ``multi_gpu``). With ``backend="gensim"`` the
        walks are kept as sequences and gensim trains on the CPU; with
        ``multi_gpu`` they are first written to a temporary Parquet directory
        under ``$TMPDIR``, and the Dask client is closed before training to
        free the GPU memory.

        Parameters
        ----------
        edge_df : cudf.DataFrame or dask_cudf.DataFrame
            Integer-encoded edge list, the output of `load_data`.
        walk_vertices : cudf.Series or None, default None
            Token ids of the start vertices. If None, walks start from every
            subject of the graph. Random walks start ``walk_number`` walks per
            vertex.

        Raises
        ------
        RuntimeError
            If no graph has been built yet (call `load_data` first).
        NotImplementedError
            If ``walk_strategy="bfs"`` is combined with ``backend="gensim"``.

        Notes
        -----
        The trained model is stored in ``self.word2vec_model``; get the
        embeddings with `transform`.

        Examples
        --------
        >>> edges = rdf2vec.load_data("example.parquet")
        >>> rdf2vec.fit(edges)
        """
        if self._walk_source_vertices is None:
            raise RuntimeError(_NO_GRAPH)
        # backend dispatch: gensim consumes walks-as-sequences via parquet
        # streaming and trains on CPU; pytorch consumes (center, context)
        # skip-gram pairs and trains on GPU via Lightning.
        if self.config.backend == "gensim":
            self._with_walk_sequences(edge_df, walk_vertices, self._fit_via_gensim)
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

    def _with_walk_sequences(self, edge_df, walk_vertices, train) -> None:
        """Generate walk sequences and pass them to `train`. Multi-GPU walks
        are written by the GPUs to a temporary parquet directory (removed
        afterwards) and passed as a `WalkParquet`."""
        import shutil
        import tempfile

        if not self.config.multi_gpu:
            train(self.walk_generation_sequences(edge_df=edge_df, walk_vertices=walk_vertices))
            return
        walk_dir = tempfile.mkdtemp(prefix="rdf2vec_walks_")
        try:
            train(self.walk_generation_sequences(
                edge_df=edge_df, walk_vertices=walk_vertices, out_dir=f"{walk_dir}/walks"
            ))
        finally:
            shutil.rmtree(walk_dir, ignore_errors=True)

    def _fit_via_gensim(self, walk_sequences) -> None:
        """Train Word2Vec via the optional gensim backend.

        Materializes the walks-as-sequences corpus to a temp parquet directory
        (gensim's streaming `WalkCorpus` iterator reads parquet files; a
        `WalkParquet` from the multi-GPU path is already on disk), tears
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
                if isinstance(walk_sequences, WalkParquet):
                    walks_parquet = walk_sequences.path  # written by the GPUs
                    _check_walk_files(walk_sequences)
                elif hasattr(walk_sequences, "to_parquet"):
                    logger.info(f"Checkpointing walk sequences to {walks_parquet}")
                    # dask_cudf.DataFrame: each worker writes its partition.
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
            strategy = _ddp_strategy(n_gpus, self.config.num_nodes)
            trainer = L.Trainer(
                max_epochs=self.config.epochs,
                log_every_n_steps=1,
                accelerator="gpu",
                precision="16-mixed",
                devices=n_gpus,
                strategy=strategy,
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
        Return the learned embeddings as a cuDF table.

        The table has one row per vocabulary token, which covers entities,
        predicates and literals, with the columns

        * ``token`` – integer id
        * ``word`` – the IRI or literal
        * ``embedding_0 … embedding_{vector_size-1}`` – float32 components

        Tokens excluded from training by ``min_count`` keep their random
        initial vector with the PyTorch backend and get a zero vector with
        the gensim backend.

        When ``generate_artifact=True``, the table is also written to
        ``./vector/embeddings_<embedding_model>.parquet`` and the vocabulary
        to ``./vector/word2idx.parquet``.

        Returns
        -------
        cudf.DataFrame
            A ``vocab_size × (vector_size + 2)`` DataFrame.

        Raises
        ------
        ValueError
            If called before `fit` (no trained model) or `load_data` (no
            vocabulary).

        Notes
        -----
        The model parameters are not changed.
        """
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
                os.makedirs("vector", exist_ok=True)
                embedding_df.to_parquet(
                    f"vector/embeddings_{self.config.embedding_model}.parquet",
                    index=False,
                )
                word2idx = self.word2idx.compute() if hasattr(self.word2idx, "compute") else self.word2idx
                word2idx.to_parquet("vector/word2idx.parquet", index=False)
            return embedding_df

    def _extract_embeddings_pytorch(self) -> cudf.DataFrame:
        """Pull `[token, word, embedding_0..n]` cudf DataFrame from the
        PyTorch trainer's in-embeddings weight matrix."""
        import cupy as cp

        model_embeddings = (
            self.word2vec_model.in_embeddings.weight.detach().cuda()
        )
        # Convert via CuPy to avoid cudf from_dlpack column-major requirement
        cp_arr = cp.from_dlpack(model_embeddings)
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
        # the library trains gensim on token ids (`int_tokens=True`)
        vectors = kv_to_token_aligned_matrix(kv, self.word2idx, int_tokens=True)
        if len(kv.index_to_key) < vectors.shape[0]:
            logger.warning(
                f"gensim KV vocab ({len(kv.index_to_key):,}) is smaller than "
                f"word2idx ({vectors.shape[0]:,}); "
                f"{vectors.shape[0] - len(kv.index_to_key):,} tokens were "
                f"filtered by min_count and will appear as zero embeddings."
            )
        cp_arr = cp.asarray(vectors)
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
        self, edge_df: cudf.DataFrame, walk_vertices: cudf.Series = None
    ) -> cudf.DataFrame:
        """
        Train the Word2Vec model and return the embeddings.

        Calls `fit` followed by `transform`.

        Parameters
        ----------
        edge_df : cudf.DataFrame or dask_cudf.DataFrame
            Integer-encoded edge list, the output of `load_data`.
        walk_vertices : cudf.Series or None, default None
            Start vertices for the walks; see `fit`.

        Returns
        -------
        cudf.DataFrame
            The embedding table returned by `transform`.

        Notes
        -----
        Exceptions raised by `fit` or `transform` propagate unchanged. The
        trained model stays in ``self.word2vec_model``.
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
        """Save the config, vocabulary and trained model to the directory
        *path* (created if needed).

        The directory holds:

        - ``config.json`` – the `RDF2VecConfig` as JSON
        - ``word2idx.parquet`` – the vocabulary (after `load_data`)
        - ``model.pt`` – the PyTorch state dict, or ``model.gensim`` – the
          gensim model with ``backend="gensim"`` (after `fit`)

        The graph is not saved; restore with `load` and call `transform`, or
        `load_data` again to train further.
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
            if self.config.backend == "gensim":
                self.word2vec_model.save(str(out / "model.gensim"))
            else:
                torch.save(self.word2vec_model.state_dict(), out / "model.pt")
        logger.info(f"Saved GPU_RDF2Vec artifacts to {out}")

    @classmethod
    def load(cls, path: str, client=None) -> "GPU_RDF2Vec":
        """Restore a GPU_RDF2Vec instance from a directory written by `save`.

        Parameters
        ----------
        path : str
            Directory written by `save`.
        client : dask.distributed.Client, optional
            Required if the saved config has ``multi_gpu=True``.

        Returns
        -------
        GPU_RDF2Vec
            An instance with config, vocabulary and model restored, ready for
            `transform`.
        """
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
        if config.backend == "gensim" and (src / "model.gensim").exists():
            from gensim.models import Word2Vec

            instance.word2vec_model = Word2Vec.load(str(src / "model.gensim"))
        elif model_path.exists():
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
