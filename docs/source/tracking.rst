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

The pipeline wraps each major step with a tracker stage context manager. Parameters and metrics
logged inside a stage are associated with that stage in the tracking backend:

- ``data_loading`` — edge count, column count, source path
- ``Literal_Handling`` — literal strategy and predicates (when configured)
- walk generation — walk strategy, depth, count, timing
- vocabulary construction — vocabulary size
- Word2Vec training — hyperparameters, loss curves, batch-size tuning results

Run metadata such as the library name and backend are recorded via tags when the pipeline starts.

Custom tracker backends
-----------------------

Implementing a custom backend means subclassing
:class:`~rdf2vecgpu.logger.base.BaseTracker` and implementing the methods relevant for your use
case (``start_pipeline``, ``stage``, ``log_params``, ``log_metrics``, ``log_artifact``, ...).
Because each method has a no-op default, you only need to override what you want to capture.
