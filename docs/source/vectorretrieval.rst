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
    # Read data from knowledge graph
    path = "data/wikidata5m/wikidata5m_kg.parquet"
    # Load data and receive edge data
    edge_data = gpu_rdf2vec_model.load_data(path)
    # Fit the model to the knowledge graph data
    gpu_rdf2vec_model.fit(edge_df=edge_data, walk_vertices=None)
    # Retrieve the learned embeddings
    embeddings = gpu_rdf2vec_model.transform()

The `transform` method returns a cudf dataframe where the keys are the entity URIs, together with 
the internal integer based ID and the embedding vectors. In case you set the **generate_artifact** 
artifact to True during the class initialization, the embeddings will also be saved to disk in the 
specified output directory as a Parquet file.
