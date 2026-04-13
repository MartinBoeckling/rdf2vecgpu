.. _dataload:

Data Loading
================

The knowledge graph data should be prepared in a file that is compatible with the package's data
load functionality. In order to load the graph, ``rdf2vecgpu`` uses two different engines with
different implications:

- **cuDF engine**: utilizes GPU memory for faster data processing. Suitable for large graphs that
  fit into GPU memory.
- **rdflib engine**: provides the ability to load graph file formats that are not directly
  supported by cuDF. However, it uses CPU memory and may be slower for large datasets.

As outlined in the :doc:`gettingstarted` guide, the engine for loading is selected based on the
provided file format. Below, we provide an overview of the supported file formats for each engine.

Supported file formats
~~~~~~~~~~~~~~~~~~~~~~~
In the following, you find an overview of the different supported file formats for both engines:

+------------------+-----------------------+-------------------------+
| File Format      | cuDF engine           | rdflib engine           |
+==================+=======================+=========================+
| N-Triples (.nt)  | Yes                   | Yes                     |
+------------------+-----------------------+-------------------------+
| N-Quads (.nq)    | Yes                   | Yes                     |
+------------------+-----------------------+-------------------------+
| Turtle (.ttl)    | No                    | Yes                     |
+------------------+-----------------------+-------------------------+
| RDF/XML (.rdf)   | No                    | Yes                     |
+------------------+-----------------------+-------------------------+
| JSON-LD (.jsonld)| No                    | Yes                     |
+------------------+-----------------------+-------------------------+
| Notation-3 (.n3) | No                    | Yes                     |
+------------------+-----------------------+-------------------------+
| Trig (.trig)     | No                    | Yes                     |
+------------------+-----------------------+-------------------------+
| CSV (.csv)       | Yes                   | No                      |
+------------------+-----------------------+-------------------------+
| Parquet (.parquet)| Yes                  | No                      |
+------------------+-----------------------+-------------------------+
| ORC (.orc)       | Yes                   | No                      |
+------------------+-----------------------+-------------------------+

For optimal performance, it is recommended to use the cuDF engine with supported file formats like
Parquet, ORC, N-Triples, or CSV. If your dataset is in a different format, consider converting it
to one of these formats for better load efficiency. The best performance is typically achieved with
Parquet files due to their columnar storage format, which is well-suited for GPU processing.

Code example
~~~~~~~~~~~~
Here is a code snippet demonstrating how to load a knowledge graph using the cuDF engine with a
Parquet file:

.. code-block:: python

   from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

   config = RDF2VecConfig(
       walk_strategy="random",
       walk_depth=4,
       walk_number=100,
       embedding_model="skipgram",
       epochs=5,
       multi_gpu=False,
   )
   gpu_rdf2vec_model = GPU_RDF2Vec(config=config)

   # Path to the triple dataset
   path = "data/wikidata5m/wikidata5m_kg.parquet"

   # Read data and receive edge data
   edge_data = gpu_rdf2vec_model.read_data(path)

Alternatively, when using a file format which is not directly supported by cuDF, this is
automatically detected and the rdflib engine is used instead:

.. code-block:: python

   edge_data = gpu_rdf2vec_model.read_data("data/wikidata5m/wikidata5m_kg.ttl")

This allows you to seamlessly load different file formats without changing the code logic.

Column mapping and reader keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``read_data`` exposes two optional arguments to adapt the reader to non-default schemas:

- ``col_map``: a mapping from your source column names to the expected ``subject``, ``predicate``,
  ``object`` (and optionally ``weights``) names.
- ``read_kwargs``: additional keyword arguments forwarded to the underlying cuDF/Dask reader
  (for example ``delimiter``, ``columns``, ``compression``).

.. code-block:: python

   edge_data = gpu_rdf2vec_model.read_data(
       "data/my_graph.csv",
       col_map={"src": "subject", "rel": "predicate", "dst": "object"},
       read_kwargs={"delimiter": "\t"},
   )

Weighted walks
~~~~~~~~~~~~~~~
When ``walk_weighted=True`` is set in the configuration, cuGraph's ``biased_random_walks`` is used
for walk generation. The input data must contain a ``weights`` column (cuGraph's standard column
name). You can use ``col_map`` to rename an existing edge-weight column accordingly:

.. code-block:: python

   config = RDF2VecConfig(walk_weighted=True, walk_strategy="random")
   gpu_rdf2vec_model = GPU_RDF2Vec(config=config)
   edge_data = gpu_rdf2vec_model.read_data(
       "data/spatial_graph.parquet",
       col_map={"distance": "weights"},
   )

Literal handling
~~~~~~~~~~~~~~~~
Knowledge graphs often contain edges whose object is a literal value (for example numeric
attributes). ``RDF2VecConfig`` exposes three parameters to handle such edges:

- ``literal_predicates``: a list of predicate strings that identify literal edges.
- ``literal_strategy``: ``"drop"`` removes literal edges from the graph (the pyRDF2Vec default),
  while ``"bin"`` discretizes the object values into bin tokens so the edge is preserved.
- ``literal_n_bins`` and ``literal_bin_strategy`` (``"quantile"`` or ``"uniform"``) control the
  binning behavior when ``literal_strategy="bin"``.

.. code-block:: python

   config = RDF2VecConfig(
       literal_predicates=["<http://example.org/hasHeight>", "<http://example.org/hasAge>"],
       literal_strategy="bin",
       literal_n_bins=5,
       literal_bin_strategy="quantile",
   )

Considerations for multi-GPU and distributed setups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Depending on the value of the ``multi_gpu`` flag in the configuration, ``read_data`` returns the
data either as a single cuDF dataframe (for single-GPU training) or as a Dask-cuDF dataframe
backed by a list of cuDF partitions (for multi-GPU training).

Depending on the framework used for the graph load, a repartition of the Dask dataframe may be
necessary to achieve the best performance for the following steps, which are influenced by the
number of GPUs available as well as the number of nodes within the cluster.
