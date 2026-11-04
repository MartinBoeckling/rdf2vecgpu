.. _tracking:

Experiment Tracking
===================

``rdf2vecgpu`` ships with a pluggable experiment tracking layer so that runs can be logged to
`MLflow <https://mlflow.org>`_, `Weights & Biases <https://wandb.ai/site/>`_, or no backend at all.
The tracker is selected via the ``tracker`` field on :class:`~rdf2vecgpu.config.RDF2VecConfig` and
instantiated by the internal :func:`~rdf2vecgpu.logger.build_tracker` factory.

Installing tracker backends
---------------------------

The tracker backends are optional dependencies. Install only the ones you need:

.. code-block:: bash

   # MLflow (uses mlflow-skinny under the hood)
   pip install "rdf2vecgpu[mlflow]"

   # Weights & Biases
   pip install "rdf2vecgpu[wandb]"

Alternatively, when working from source with ``uv``:

.. code-block:: bash

   uv sync --extra mlflow
   uv sync --extra wandb

Selecting a tracker
-------------------

.. code-block:: python

   from rdf2vecgpu import GPU_RDF2Vec, RDF2VecConfig

   config = RDF2VecConfig(
       walk_strategy="random",
       walk_depth=4,
       walk_number=100,
       embedding_model="skipgram",
       epochs=5,
       tracker="mlflow",  # "none" (default), "mlflow", or "wandb"
       tracker_run_name="wikidata5m-baseline",
       tracker_kwargs={
           "mlflow": {
               "tracking_uri": "http://mlflow.internal:5000",
               "experiment_name": "rdf2vecgpu",
           },
       },
   )
   model = GPU_RDF2Vec(config=config)

For Weights & Biases, use ``tracker="wandb"`` and provide backend-specific kwargs under the
``"wandb"`` key (for example ``project``, ``entity``, ``group``).

.. note::
   When ``tracker="none"`` (the default), a :class:`~rdf2vecgpu.logger.base.NoOpTracker` is used.
   All tracker calls become no-ops, so the pipeline behaves identically to an untraced run.

Pipeline stages
---------------

The pipeline wraps each major step with a tracker stage context manager. With MLflow every stage
is a nested run (so its start and end time give the stage duration); with Weights & Biases the
start and end of each stage are logged as ``stage/start`` and ``stage/end`` events. The stages and
the parameters logged in them are:

- ``data_loading`` — source path, number of rows and columns
- ``Literal_Handling`` — literal strategy and predicates (only when ``literal_predicates`` is set)
- ``Word2idx_Generation`` — vocabulary size
- ``Graph_Construction`` — graph size and degree statistics
- ``Walk_Generation`` — walk strategy, depth, number, weighting and ``min_count``
- ``Word2Vec_Training`` — training hyperparameters; with MLflow and the PyTorch backend,
  ``mlflow.pytorch.autolog`` also records the Lightning training metrics, and the trained model
  is logged as an artifact
- ``Embedding_Extraction`` — extraction of the embedding table in ``transform``

Run metadata such as the library name and backend are recorded via tags when the pipeline starts.

Custom tracker backends
-----------------------

Implementing a custom backend means subclassing
:class:`~rdf2vecgpu.logger.base.BaseTracker` and implementing the methods relevant for your use
case (``start_pipeline``, ``stage``, ``log_params``, ``log_metrics``, ``log_artifact``, ...).
Because each method has a no-op default, you only need to override what you want to capture.
