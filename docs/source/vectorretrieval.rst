.. _vectorretrieval:
Retrieval of embeddings
=========================
After training RDF2Vec embeddings using the `gpuRDF2vec` package, you can retrieve the vector 
representations for all entities used within the knowledge graph. Similarly to the GPU-based 
training process, the retrieval of embeddings is also optimized for performance by building on top
of dlpack to extract the vectors directly from GPU memory. This allows in general the possibility
to handle large-scale knowledge graphs efficiently.

The following example demonstrates how to perform this retrieval process:

.. code-block:: python

    from gpuRDF2vec import GPU_RDF2vec

    # Initialize the GPU_RDF2vec object with the path to your knowledge graph
    rdf2vec = GPU_RDF2vec(kg_path="path/to/your/knowledge_graph.ttl")

    # Train the RDF2Vec embeddings
    rdf2vec.train(embedding_size=128, epochs=10)

    # Retrieve the embeddings for all entities
    embeddings = rdf2vec.transform()

The `transform` method returns a cudf dataframe where the keys are the entity URIs, together with 
the internal integer based ID and the embedding vectors. In case you set the **generate_artifact** 
artifact to True during the class initialization, the embeddings will also be saved to disk in the 
specified output directory as a Parquet file.
