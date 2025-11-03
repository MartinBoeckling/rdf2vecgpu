.. _gettingstarted: 
Getting Started
=================
The starting point for using rdf2vecgpu is to install the package as described in :doc:`installation`. 
After installation, you can follow these steps to get started with generating RDF2Vec embeddings 
using GPU acceleration.

The overall framework design is oriented using similar abstractions as with scikit-learn. The main class
to interact with is `GPU_RDF2Vec` which provides methods for loading data, fitting the model, and transforming
the data into embeddings.

The first step of the process is to instantiate the `GPU_RDF2Vec` class with the desired parameters for walk strategy,
embedding model, and training settings. The following code snippet demonstrates how to initialize the model, load a knowledge graph from a file,
fit the Word2Vec model, and transform the data into embeddings.

The `fit_transform` method combines the fitting of the Word2Vec model and the transformation of 
the data into embeddings in one step. Both can also be independently called using the `fit` and `transform` methods.

Basic usage 
~~~~~~~~~~~~
.. code-block:: python

   from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec
    # Instantiate the gpu RDF2Vec library settings
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
    # Fit the Word2Vec model and transform the dataset to an embedding
    embeddings = gpu_rdf2vec_model.fit_transform(edge_df=edge_data, walk_vertices=None)
    # Write embedding to file format. Return format is a cuDf dataframe
    embeddings.to_parquet("data/wikidata5m/wikidata5m_embeddings.parquet", index=False)

Outlook
~~~~~~~~~~~~
Currently, the package supports the overall workflow following the scikit-learn paradigm. 
In the future releases we will provide more fine granular interfaces to allow users to
customize the different steps based on the specific use case. In addition, this will generally
benefit the **multi-GPU** support and distributed training capabilities in order to reduce the task
graph of Dask for very large graphs by allowing users to persist data between the steps.