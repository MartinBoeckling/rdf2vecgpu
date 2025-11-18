from typing import Optional
from cugraph import Graph
import cugraph
import lightning as L
import torch
import dask
from torch.utils.dlpack import to_dlpack
from lightning.pytorch.tuner import Tuner
from .helper.functions import _generate_vocab, cudf_to_torch_tensor, torch_to_cudf
from .embedders.word2vec import SkipGram, CBOW, OrderAwareSkipgram, OrderAwareCBOW
from .embedders.word2vec_loader import (
    SkipGramDataModule,
    CBOWDataModule,
    OrderAwareSkipGramDataModule,
    OrderAwareCBOWDataModule,
)
from .reader.kg_file_reader import KGFileReader
from .corpus.walk_corpus import single_gpu_walk_corpus, multi_gpu_walk_corpus
from .logger import BaseTracker, NoOpTracker, build_tracker
import cudf
import dask.dataframe as dd
from loguru import logger
from .config import RDF2VecConfig


class GPU_RDF2Vec:
    def __init__(
        self,
        config: RDF2VecConfig,
        client: Optional[dask.distributed.Client] = None,
        **kwargs,
    ):  # Add client parameter
        """GPU‑accelerated implementation of the RDF2Vec pipeline.

        This class wraps every step that is necessary to obtain dense vector
        representations for entities in a (potentially very large) knowledge
        graph on the GPU:

        1. **Load the triples** into a cuGraph `Graph`, optionally persisting
        intermediate artefacts.
        2. **Generate random walks** (or, in the future, BFS walks) that serve as a
        textual corpus.
        3. **Train a Word2Vec model** (currently Skip‑gram, CBOW forthcoming) on
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
        else:
            self.client = None
        self._validate_environment()
        # Initialize the cugraph graph
        self.knowledge_graph = Graph(directed=True)
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

    def generate_vocab(self, kg_data):
        with self.tracker.stage("Word2idx_Generation"):
            word2idx = _generate_vocab(kg_data, self.config.multi_gpu)
            self.tracker.log_params({"vocab_size": word2idx.shape[0]})
        self.word2idx = word2idx
        return word2idx

    def construct_graph(self, kg_data):
        with self.tracker.stage("Graph_Construction"):
            if self.config.multi_gpu:
                kg_data = kg_data[["subject", "predicate", "object"]]
                if cugraph.dask.comms.is_initialized():
                    self.knowledge_graph.from_dask_cudf_edgelist(
                        kg_data,
                        source="subject",
                        destination="object",
                        edge_attr="predicate",
                        renumber=False,
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
                    kg_data,
                    source="subject",
                    destination="object",
                    edge_attr="predicate",
                    renumber=False,
                )
            degree = self.knowledge_graph.degree()
            number_nodes = self.knowledge_graph.number_of_vertices()
            number_edges = self.knowledge_graph.number_of_edges()
            self.tracker.log_metrics(
                {
                    "number_nodes": number_nodes,
                    "number_edges": number_edges,
                    "average_degree": degree.mean(),
                    "min_degree": degree.min(),
                    "max_degree": degree.max(),
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
        word2idx = self.generate_vocab(kg_data)
        self.construct_graph(kg_data)

    def walk_generation(
        self, edge_df: cudf.DataFrame, walk_vertices: cudf.Series = None
    ) -> cudf.DataFrame:
        with self.tracker.stage("Walk_Generation"):
            if self.config.multi_gpu:
                walk_instance = multi_gpu_walk_corpus(
                    self.knowledge_graph,
                    self.config.window_size,
                )
            else:
                walk_instance = single_gpu_walk_corpus(
                    self.knowledge_graph,
                    self.config.window_size,
                )
            if self.config.walk_strategy == "random":
                if walk_vertices is None:
                    walk_vertices = self.knowledge_graph.nodes()
                walk_vertices = walk_vertices.repeat(self.config.walk_number)
                walk_corpus = walk_instance.random_walk(
                    edge_df=edge_df,
                    walk_vertices=walk_vertices,
                    walk_depth=self.config.walk_depth,
                    random_state=self.config.random_state,
                    word2vec_model=self.config.embedding_model,
                    min_count=self.config.min_count,
                )
            elif self.config.walk_strategy == "bfs":
                if walk_vertices is None:
                    walk_vertices = self.knowledge_graph.nodes()
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
                    "min_count": self.config.min_count,
                }
            )

        return walk_corpus

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
        walk_corpus = self.walk_generation(
            edge_df=edge_df,
            walk_vertices=walk_vertices,
        )
        center_tensor = torch.utils.dlpack.from_dlpack(walk_corpus, "center")
        context_tensor = torch.utils.dlpack.from_dlpack(walk_corpus, "context")
        with self.tracker.stage("Word2Vec_Training"):
            if self.config.embedding_model == "skipgram":
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
                        else round(len(context_tensor) / (self.config.cpu_count))
                    ),
                )

            elif self.config.embedding_model == "cbow":
                word2vec_model = CBOW(
                    vocab_size=self.word2idx.shape[0],
                    embedding_dim=self.config.vector_size,
                    learning_rate=self.config.learning_rate,
                )
                # TODO: Optimize context tensor creation
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
                        else round(len(context_tensor) / (self.config.cpu_count))
                    ),
                )

            else:
                logger.error(
                    f"Unsupported embedding model: {self.config.embedding_model}. Please choose either 'skipgram' or 'cbow'."
                )
                raise ValueError(
                    f"Unsupported embedding model: {self.config.embedding_model}. Please choose either 'skipgram' or 'cbow'."
                )
            if self.config.reproducible:
                logger.info(
                    "Setting up reproducible training, might increase training time."
                )
                L.seed_everything(self.config.random_state, workers=True)
            self.tracker.log_pytorch()
            trainer = L.Trainer(
                max_epochs=self.config.epochs,
                log_every_n_steps=1,
                accelerator="gpu",
                precision=16,
                devices="auto",
                num_nodes=self.config.num_nodes,
            )
            if self.config.tune_batch_size:
                tuner = Tuner(trainer)
                tuner.scale_batch_size(
                    word2vec_model,
                    mode="power",
                    datamodule=datamodule,
                    steps_per_trial=1,
                    init_val=round(len(context_tensor) / (self.config.cpu_count * 2)),
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
        # Check if model is fitted and word2idx is available
        with self.tracker.stage("Embedding_Extraction"):
            if self.word2vec_model is not None and self.word2idx is not None:
                model_embeddings = self.word2vec_model.in_embeddings.weight
                model_embeddings_df = torch_to_cudf(
                    model_embeddings.T, self.config.multi_gpu
                ).T
                model_embeddings_df.columns = model_embeddings_df.columns.astype(str)
                model_embeddings_df = model_embeddings_df.add_prefix("embedding_")
                embedding_df = cudf.concat([self.word2idx, model_embeddings_df], axis=1)
                if self.config.generate_artifact:
                    embedding_df.to_parquet(
                        f"vector/embeddings_{self.word2vec_model}.parquet", index=False
                    )
                return embedding_df
            else:
                raise ValueError(
                    "The transform method is not possible to call without a fitted model or a generated word2idx setup."
                    "Please call the 'fit' method first or the 'load_data' method to generate the word2idx setup."
                )

    def fit_transform(
        self, edge_df: cudf.DataFrame, walk_vertices: cudf.DataFrame
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
        if self.client is not None:
            self.client.close()

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
        if self.config.multi_gpu and self.client is None:
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
        if self.config.multi_gpu and torch.cuda.device_count() < 2:
            logger.warning(
                f"multi_gpu=True but torch reports {torch.cuda.device_count()} visible GPU on this process."
            )
