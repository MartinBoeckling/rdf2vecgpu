gpuRDF2vec documentation
========================

gpuRDF2Vec is a scalable GPU-based implementation of RDF2Vec embeddings for large and dense Knowledge Graphs.

.. image:: _static/img/rdf2vecgpu_logo.png
   :alt: RDF2VecGPU Image
   :align: center
   :width: 600px

.. note::
   Licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.

How it works
------------

The data stays on the GPU from loading to training, except for RDF formats that are parsed with
rdflib; only the gensim backend trains on the CPU. The design and the benchmarks against CPU-based
RDF2Vec implementations are described in the
`ISWC 2025 paper <https://doi.org/10.1007/978-3-032-09530-5_14>`__.

1. **Loading** (``load_data``): the triples are read with cuDF (Dask-cuDF with ``multi_gpu=True``).
   Every IRI and literal becomes an integer token, and the encoded triples form a directed cuGraph
   ``MultiGraph``, so parallel edges with different predicates are kept.

2. **Walks** (``fit``): every start vertex (by default every subject) is repeated ``walk_number``
   times and the walks are generated in bulk by cuGraph: ``uniform_random_walks``, or
   ``biased_random_walks`` with ``walk_weighted=True``. The predicate of every step is recovered
   from the edge that was actually traversed. On one GPU the walks are generated in chunks of
   up to 25 M walks and kept as int32 token matrices. BFS walks run one cuGraph BFS per start
   vertex and give one walk per root-to-leaf path, so their cost grows with the size of the graph.
   With ``multi_gpu=True``, the walks are generated on all GPUs of the Dask cluster.
   ``prepare_corpus`` writes the walks to Parquet instead, and ``fit_from_walks`` trains from those
   files later, without the graph.

3. **Training data** (``backend="pytorch"``): the walks stay in GPU memory as sequences. Every step
   takes a shuffled batch of walks, drops frequent tokens (``sample``, as in word2vec) and builds the
   (center, context) pairs of that batch on the GPU. Steps are sized to hold about ``batch_size``
   pairs (default 32768) after subsampling and walks that end early. With ``multi_gpu=True``, the
   walks are written to Parquet, the Dask client is closed, and each GPU reads its own block.

4. **Training**: skip-gram, CBOW or order-aware skip-gram with negative sampling, trained with
   PyTorch Lightning the way word2vec trains: summed loss, sparse SGD with a learning rate that
   decays linearly from ``learning_rate`` to ``min_learning_rate``, and negatives drawn by token
   frequency^0.75. With ``multi_gpu=True``, Lightning trains with DDP and averages the gradients
   every step. With ``backend="gensim"``, the walks are streamed from Parquet into gensim's
   Word2Vec on the CPU instead.

5. **Output** (``transform``): the embedding matrix is copied into a cuDF table with one row per
   token (``token``, ``word``, ``embedding_*``), optionally written to Parquet.

Report Issues and Bugs
----------------------

Please open an issue with the label **Bug** and provide using the following template under the `Github issue page <https://github.com/MartinBoeckling/rdf2vecgpu/issues>`__:

- **Environment**: OS, Python, CUDA, PyTorch, cuDF versions
- **Reproduction steps**: Code or CLI input
- **Dataset**: Format & size
- **Observed behavior** vs **expected behavior**
- **Error logs** or stack traces

We aim to respond within **3 business days**. For fixes, open a PR referencing the issue.

License
-------

This project is licensed under the MIT License.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Installation

   installation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: gpuRDF2vec Usage

   gettingstarted
   configuration
   dataload
   training
   vectorretrieval
   tracking

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Benchmarks

   performance

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API reference

   modules


Indices and tables
------------------

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
