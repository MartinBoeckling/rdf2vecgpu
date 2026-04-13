.. _vectorretrieval:

Retrieval of embeddings
=========================
After training RDF2Vec embeddings using the ``gpuRDF2vec`` package, you can retrieve the vector
representations for all entities used within the knowledge graph. Similarly to the GPU-based
training process, the retrieval of embeddings is also optimized for performance by building on top
of DLPack to extract the vectors directly from GPU memory. This allows you to handle large-scale
knowledge graphs efficiently.

The following example demonstrates how to perform this retrieval process:

.. code-block:: python

   from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

   # Initialize the GPU_RDF2Vec pipeline
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

   # Train the RDF2Vec embeddings
   gpu_rdf2vec_model.fit(edge_df=edge_data, walk_vertices=None)

   # Retrieve the embeddings for all entities
   embeddings = gpu_rdf2vec_model.transform()

The ``transform`` method returns a cuDF dataframe where the keys are the entity URIs, together with
the internal integer-based ID and the embedding vectors. If you set ``generate_artifact=True``
during the configuration, the embeddings will also be saved to disk in the specified output
directory as a Parquet file.
