.. _dataload: 
Data Loading
================

The knowledge graph data should be prepared in a file that is compatible with the package's data load functionality. In order to load the graph, we use two different engines that have different implications:
- **cuDF Engine**: Utilizes GPU memory for faster data processing. Suitable for large graphs that fit into GPU memory.
- **rdflib engine**: Provides possibility to load graph file formats that are not directly supported by cuDF. However, it uses CPU memory and may be slower for large datasets.

As outlined in the `getting started guide <gettingstarted>`_, the engine for loading is selected based on the provided file format. Below, we provide an overview of the supported file formats for each engine.

Supported File Formats
~~~~~~~~~~~~~~~~~~~~~~~
In the following, you find an overview of the different supported file formats for both engines:

+------------------+-------------------------------+-------------------------------+
| File Format      | cuDF Engine Supported         | rdflib Engine Supported       |
+==================+===============================+===============================+
| N-Triples (.nt)  | Yes                           | Yes                           |
+------------------+-------------------------------+-------------------------------+
| Turtle (.ttl)    | No                            | Yes                           |
+------------------+-------------------------------+-------------------------------+
| RDF/XML (.rdf)   | No                            | Yes                           |
+------------------+-------------------------------+-------------------------------+
| JSON-LD (.jsonld)| No                            | Yes                           |
+------------------+-------------------------------+-------------------------------+
| Notation-3 (.n3) | No                            | Yes                           |
+------------------+-------------------------------+-------------------------------+
| Trig (.trig)     | No                            | Yes                           |
+------------------+-------------------------------+-------------------------------+
| CSV (.csv)       | Yes                           | No                            |
+------------------+-------------------------------+-------------------------------+
| Parquet (.parquet) | Yes                         | No                            |
+------------------+-------------------------------+-------------------------------+

For optimal performance, it is recommended to use the cuDF engine with supported file formats like N-Triples, CSV, or Parquet. If your dataset is in a different format, consider converting it to one of these formats for better load efficiency. The best performance is achieved with Parquet files due to their columnar storage format, which is well-suited for GPU processing.

Code example
~~~~~~~~~~~~
Here is a code snippet demonstrating how to load a knowledge graph using the cuDF engine with an N-Triples file:

.. code-block:: python

   gpu_rdf2vec_model = GPU_RDF2Vec(
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
        cpu_count=20
    )
    # Path to the triple dataset
    path = "data/wikidata5m/wikidata5m_kg.parquet"
    # Load data and receive edge data
    edge_data = gpu_rdf2vec_model.load_data(path)

   # Now you can use the knowledge_graph for further processing

Alternatively, when using a file format which is not directly supported by cuDF, this is automatically detected and the rdflib engine is used instead:

.. code-block:: python

   gpu_rdf2vec_model = GPU_RDF2Vec(
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
        cpu_count=20
    )
    # Path to the triple dataset
    path = "data/wikidata5m/wikidata5m_kg.ttl"
    # Load data and receive edge data
    edge_data = gpu_rdf2vec_model.load_data(path)

   # Now you can use the knowledge_graph for further processing

This allows to seemlessly load different file formats without changing the code logic.


Considerations for Multi-GPU and Distributed Setups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Depending on the selection of the parameter of the `multi_gpu` flag during model initialization, 
the method `load_data` will provide the data either as a single cuDF dataframe (for single-GPU training) 
or as a Dask dataframe that is provided as a list of cuDF dataframes (for multi-GPU training). 

Based on the framework used for the graph load, a reparatition of the Dask dataframe is necessary 
in order to achieve the best performance for the following steps that are influenced by the number 
of GPUs available as well as the number of nodes within the cluster.