.. _gettingstarted:

Getting Started
=================
The starting point for using ``rdf2vecgpu`` is to install the package as described in :doc:`installation`.
After installation, you can follow these steps to get started with generating RDF2Vec embeddings
using GPU acceleration.

The overall framework design is oriented using similar abstractions as with scikit-learn. The main class
to interact with is :class:`~rdf2vecgpu.gpu_rdf2vec.GPU_RDF2Vec` which provides methods for reading data,
fitting the model, and transforming the data into embeddings. All hyperparameters are bundled in a
:class:`~rdf2vecgpu.config.RDF2VecConfig` object — see :doc:`configuration` for the full parameter reference.

The first step is to instantiate ``GPU_RDF2Vec`` with an ``RDF2VecConfig`` (or equivalent keyword
arguments), read a knowledge graph from a file, then fit and transform. The ``fit_transform`` method
combines fitting the Word2Vec model and returning the embeddings in one step; both can also be called
independently via ``fit`` and ``transform``.

Basic usage
~~~~~~~~~~~~
.. code-block:: python

   from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

   # Bundle all hyperparameters in a config object
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

   # Instantiate the pipeline
   gpu_rdf2vec_model = GPU_RDF2Vec(config=config)

   # Path to the triple dataset
   path = "data/wikidata5m/wikidata5m_kg.parquet"

   # Read data and receive edge data
   edge_data = gpu_rdf2vec_model.read_data(path)

   # Fit the Word2Vec model and transform the dataset to an embedding
   embeddings = gpu_rdf2vec_model.fit_transform(edge_df=edge_data, walk_vertices=None)

   # Write embedding to file format. Return format is a cuDF dataframe
   embeddings.to_parquet("data/wikidata5m/wikidata5m_embeddings.parquet", index=False)

As a shorthand, keyword arguments can be passed directly to ``GPU_RDF2Vec`` and they will be forwarded
to ``RDF2VecConfig`` internally:

.. code-block:: python

   gpu_rdf2vec_model = GPU_RDF2Vec(
       walk_strategy="random",
       walk_depth=4,
       walk_number=100,
       embedding_model="skipgram",
       epochs=5,
   )

Outlook
~~~~~~~~~~~~
Currently, the package supports the overall workflow following the scikit-learn paradigm.
In the future releases we will provide more fine granular interfaces to allow users to
customize the different steps based on the specific use case. In addition, this will generally
benefit the **multi-GPU** support and distributed training capabilities in order to reduce the task
graph of Dask for very large graphs by allowing users to persist data between the steps.
