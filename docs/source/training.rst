.. _training:

Training RDF2vec with gpuRDF2vec
=================================
The training process of RDF2vec embeddings using the ``gpuRDF2vec`` package involves several steps
that happen internally and leverage the GPU acceleration capabilities of the package.
The overall training of the embedding model happens by calling the ``fit`` method of the
:class:`~rdf2vecgpu.gpu_rdf2vec.GPU_RDF2Vec` class.
Below, we outline the main steps that are performed during the training process:

1. **Data Reading**: ``read_data`` loads the triples into a cuDF (single-GPU) or Dask-cuDF
   (multi-GPU) dataframe using :class:`~rdf2vecgpu.reader.kg_file_reader.KGFileReader`.

2. **Walk Extraction**: Based on the selected ``walk_strategy`` (``random`` or ``bfs``), the package
   generates walks from the knowledge graph. This step is performed entirely on the GPU using cuGraph
   for efficient graph traversal and walk generation. When ``walk_weighted=True``, cuGraph's
   ``biased_random_walks`` is used and the input must contain a ``weights`` column.

3. **Data Preparation**: The generated walks are converted into a format suitable for the
   embedding model using cuDF dataframes, which are handed off to PyTorch tensors via DLPack to
   avoid CPU bottlenecks.

4. **Embedding Training**: The Word2Vec model is trained on the prepared walks. The package uses
   an optimized implementation of Word2Vec that leverages GPU acceleration for faster training times
   and allows scaling across different nodes and GPUs via PyTorch Lightning and Dask.

5. **Model Saving**: After training, the learned embeddings can be saved to disk for later use.

Here is an example code snippet demonstrating how to train RDF2vec embeddings using the
``gpuRDF2vec`` package:

.. code-block:: python

   from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

   # Initialize the GPU_RDF2Vec model with desired parameters
   config = RDF2VecConfig(
       walk_strategy="random",
       walk_depth=4,
       walk_number=100,
       embedding_model="skipgram",
       epochs=5,
       batch_size=None,
       vector_size=100,
       window_size=5,
       min_count=1,
       learning_rate=0.01,
       negative_samples=5,
       random_state=42,
       reproducible=False,
       multi_gpu=False,
       generate_artifact=False,
       cpu_count=20,
   )
   gpu_rdf2vec_model = GPU_RDF2Vec(config=config)

   # Read the knowledge graph
   edge_data = gpu_rdf2vec_model.read_data("data/wikidata5m/wikidata5m_kg.parquet")

   # Fit the model to the knowledge graph data
   gpu_rdf2vec_model.fit(edge_df=edge_data, walk_vertices=None)


Pipeline stages and experiment tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Internally, the pipeline is divided into stages wrapped by the tracker's context manager, for
example ``data_loading``, ``Literal_Handling``, walk generation, vocabulary construction, and
training. When a tracker backend is configured via ``config.tracker`` (``"mlflow"`` or ``"wandb"``),
parameters and metrics for each stage are logged to the selected experiment tracking backend.
See :doc:`tracking` for details on the available backends, required extras, and what is captured
at each stage.
