.. _training:

Training RDF2vec with gpuRDF2vec
=================================
The training process of RDF2vec embeddings using the `gpuRDF2vec` package involves several steps that
happen internally which use the GPU acceleration capabilities of the package. 
The overall training of the embedding model happens by calling the `fit` method of the `GPU_RDF2Vec` class.
Below, we outline the main steps that are performed during the training process:

1. **Walk Extraction**: Based on the selected walk strategy (random or BFS), the package generates 
walks from the knowledge graph. This step is performed entirely on the GPU using cuGraph for 
efficient graph traversal and walk generation.

2. **Data Preparation**: The generated walks are then prepared for training the Word2Vec model. This involves converting the walks into a format 
suitable for the embedding model, which is also done on the GPU using cuDF dataframes which transforms
the data into PyTorch tensors via DLPack to avoid CPU bottlenecks.

3. **Embedding Training**: The Word2Vec model is trained using the prepared walks. 
The package uses an optimized implementation of Word2Vec that leverages GPU acceleration for 
faster training times and allows scaling accross different nodes and GPUs.

4. **Model Saving**: After training, the learned embeddings can be saved to disk for later use.
Here is an example code snippet demonstrating how to train RDF2vec embeddings using the `gpuRDF2vec` package:

.. code-block:: python

   from gpu_rdf2vec import GPU_RDF2Vec

   # Initialize the GPU_RDF2Vec model with desired parameters
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

   # Fit the model to the knowledge graph data
   gpu_rdf2vec_model.fit(
       graph_path="path/to/knowledge_graph.nt",
       graph_format="nt",
       engine="cudf"
   )