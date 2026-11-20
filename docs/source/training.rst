.. _training:

Training RDF2vec with gpuRDF2vec
=================================
The training process of RDF2vec embeddings using the ``gpuRDF2vec`` package involves several steps
that happen internally and leverage the GPU acceleration capabilities of the package.
The overall training of the embedding model happens by calling the ``fit`` method of the
:class:`~rdf2vecgpu.gpu_rdf2vec.GPU_RDF2Vec` class.
Below, we outline the main steps that are performed during the training process:

1. **Data Loading** (``load_data``, before ``fit``): reads the triples into a cuDF (single-GPU)
   or Dask-cuDF (multi-GPU) dataframe using
   :class:`~rdf2vecgpu.reader.kg_file_reader.KGFileReader`, encodes them as integer tokens and
   builds the cuGraph graph.

2. **Walk Extraction**: Based on the selected ``walk_strategy`` (``random`` or ``bfs``), the package
   generates walks from the knowledge graph. This step is performed entirely on the GPU using cuGraph
   for efficient graph traversal and walk generation. When ``walk_weighted=True``, cuGraph's
   ``biased_random_walks`` is used and the input must contain a ``weights`` column.

3. **Data Preparation**: With ``backend="pytorch"`` the walks stay in GPU memory as int32 token
   matrices. Every training step takes a shuffled batch of walks, drops frequent tokens
   (``sample``) and builds that batch's (center, context) pairs on the GPU, so the full pair
   list is never materialised. With ``backend="gensim"`` the walks are written to Parquet and
   streamed to gensim.

4. **Embedding Training**: skip-gram, CBOW or order-aware skip-gram with negative sampling, trained
   with PyTorch Lightning like word2vec (summed loss, sparse SGD, learning rate decaying linearly
   from ``learning_rate`` to ``min_learning_rate``), on one GPU or with DDP across GPUs and nodes.

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
       negative_samples=5,
       random_state=42,
       reproducible=False,
       multi_gpu=False,
       generate_artifact=False,
       cpu_count=20,
   )
   gpu_rdf2vec_model = GPU_RDF2Vec(config=config)

   # Read the file, build the vocabulary and the graph
   edge_data = gpu_rdf2vec_model.load_data("data/wikidata5m/wikidata5m_kg.parquet")

   # Fit the model to the knowledge graph data
   gpu_rdf2vec_model.fit(edge_df=edge_data, walk_vertices=None)


Batch size
~~~~~~~~~~
``batch_size`` (default 32768) is the number of (center, context) pairs per training step. On one
GPU, the number of walks per step is chosen before training so that a step holds about
``batch_size`` pairs after subsampling and after walks that end early, whatever the walk depth.
Under DDP every rank must run the same number of steps, so the walks per step are sized for
full-length walks instead, and steps hold fewer pairs than ``batch_size``. A step never uses more
than 1/100 of the training pairs (per GPU), so every epoch has at least 100 steps.

Larger batches train faster with fewer steps, at some cost in embedding quality: in our checks on
FB15k-237, doubling the batch to 65k pairs lowered quality by about 1% at depth 4. ``shrink_window=True`` draws each window size
uniformly from 1..``window_size`` as word2vec does: about a third fewer pairs and faster training,
at somewhat lower embedding quality.

Generating walks once and training later
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``prepare_corpus`` writes the walks and the vocabulary to a directory, and ``fit_from_walks``
trains from them without the graph and without ``load_data`` (as jRDF2Vec's ``-onlyWalks`` and
``-onlyTraining``). Walks can be generated once and trained on with several configurations:

.. code-block:: python

   from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

   walker = GPU_RDF2Vec(config=RDF2VecConfig(walk_depth=4, walk_number=100))
   edge_data = walker.load_data("data/fb15k/fb15k_kg.parquet")
   walks_path, vocab_path = walker.prepare_corpus(edge_data, "corpus/fb15k")

   # later, possibly in another process
   model = GPU_RDF2Vec(config=RDF2VecConfig(embedding_model="order_aware_skipgram", epochs=5))
   model.fit_from_walks(walks_path, vocab_path)
   embeddings = model.transform()

``prepare_corpus`` supports random walks only.

Multi-GPU training
~~~~~~~~~~~~~~~~~~
With ``multi_gpu=True`` the walks are written to a temporary parquet directory
(``tempfile.mkdtemp``, i.e. under ``$TMPDIR`` or ``/tmp``) before training. Large
corpora can take hundreds of GB, so point ``TMPDIR`` at a disk with enough space.

* ``backend="gensim"`` trains in one process on the node of the Dask client and
  reads the walk files the Dask workers wrote, so all workers must run on that
  node (as with a ``LocalCUDACluster``). Training stops with an error if walk
  files are missing.
* ``backend="pytorch"`` trains with DDP. On one node, Lightning starts one
  process per GPU itself by running the training script again; when SLURM
  starts one task per GPU (``srun --ntasks-per-node=<GPUs>``), the launch is
  left to SLURM. Each started process repeats the script, including data
  loading and walk generation, so for multi-GPU training it is usually better
  to write the walks with ``prepare_corpus`` first and train with
  ``fit_from_walks`` and ``multi_gpu=True``, which needs no Dask client.
* Multi-GPU PyTorch CBOW is supported on random walks; on BFS walks it needs
  ``backend="gensim"``.

Pipeline stages and experiment tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Internally, the pipeline is divided into stages wrapped by the tracker's context manager:
``data_loading``, ``Literal_Handling``, ``Word2idx_Generation``, ``Graph_Construction``,
``Walk_Generation``, ``Word2Vec_Training`` and ``Embedding_Extraction``. When a tracker backend is configured via ``config.tracker`` (``"mlflow"`` or ``"wandb"``),
parameters and metrics for each stage are logged to the selected experiment tracking backend.
See :doc:`tracking` for details on the available backends, required extras, and what is captured
at each stage.
