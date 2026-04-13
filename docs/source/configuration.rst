.. _configuration:

Configuration reference
=======================

All hyperparameters for ``rdf2vecgpu`` are centralized in the
:class:`~rdf2vecgpu.config.RDF2VecConfig` Pydantic model. This page provides the full parameter
reference, grouped by concern:

- **Walk generation**: walk strategy, depth, number of walks per vertex, weighted walks.
- **Embedding model**: Word2Vec variant, vector size, window, negative sampling, learning rate.
- **Training**: epochs, batch size, batch-size tuning, reproducibility, CPU worker count.
- **Execution**: single- vs. multi-GPU, number of cluster nodes, artifact export.
- **Literal handling**: predicate filtering, binning strategy, bin count.
- **Experiment tracking**: backend selection, run name, backend-specific kwargs.

Usage
-----

.. code-block:: python

   from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

   config = RDF2VecConfig(
       walk_strategy="random",
       walk_depth=4,
       walk_number=100,
       embedding_model="skipgram",
       epochs=5,
       learning_rate=0.01,
       multi_gpu=False,
       tracker="none",
   )
   model = GPU_RDF2Vec(config=config)

As a shorthand, keyword arguments passed directly to ``GPU_RDF2Vec`` are forwarded to
``RDF2VecConfig``:

.. code-block:: python

   model = GPU_RDF2Vec(walk_strategy="random", walk_depth=4, epochs=5)

Full parameter reference
------------------------

.. autoclass:: rdf2vecgpu.config.RDF2VecConfig
   :members:
   :show-inheritance:
   :undoc-members:
