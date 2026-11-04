.. _installation_guide:

Installation Guide
==================

Overview
--------

``rdf2vecgpu`` is distributed on PyPI. ``pip install rdf2vecgpu`` installs the package together
with its GPU stack: PyTorch, PyTorch Lightning and the RAPIDS libraries cuDF, Dask-cuDF and
cuGraph (CUDA 12 builds). A CUDA GPU is required; the library raises an ``EnvironmentError`` at
start-up if none is available.

Prerequisites
-------------

- Python 3.12 or newer
- Linux (on Windows, use WSL2; the RAPIDS pip wheels are Linux-only)
- An NVIDIA GPU with a CUDA 12.x driver

macOS is not supported, because CUDA is not available there.

Quick install (PyPI)
--------------------

Create and activate a virtual environment (recommended), then install:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

   # Upgrade packaging tools
   python -m pip install -U pip

   # Install rdf2vecgpu from PyPI
   python -m pip install -U rdf2vecgpu

Optional extras
---------------

``rdf2vecgpu`` ships several optional dependency groups. Install the ones you need:

.. code-block:: bash

   # CPU Word2Vec trainer (RDF2VecConfig(backend="gensim"))
   python -m pip install -U "rdf2vecgpu[gensim]"

   # Experiment tracking backends
   python -m pip install -U "rdf2vecgpu[mlflow]"
   python -m pip install -U "rdf2vecgpu[wandb]"

   # Test dependencies
   python -m pip install -U "rdf2vecgpu[test]"

When working from source with ``uv``, the equivalent commands are:

.. code-block:: bash

   uv sync --extra gensim
   uv sync --extra mlflow
   uv sync --extra wandb
   uv sync --extra test

PyTorch build
-------------

PyTorch is installed as a dependency. If you need a specific PyTorch build for your CUDA
version, install it first with the official selector (https://pytorch.org/get-started/locally/),
then install ``rdf2vecgpu``. Check that PyTorch sees the GPU:

.. code-block:: python

   import torch
   print("CUDA available:", torch.cuda.is_available())

Conda (optional)
----------------

If you prefer Conda for environment management:

.. code-block:: bash

   conda create -n rdf2vecgpu python=3.12 -y
   conda activate rdf2vecgpu
   python -m pip install -U pip
   python -m pip install -U rdf2vecgpu

Install from source (this repository)
-------------------------------------

If you are working with the sources in this repo:

.. code-block:: bash

   # From the project root
   python -m venv .venv
   source .venv/bin/activate

   python -m pip install -U pip
   python -m pip install -e ".[test]"

Multi-GPU setup
---------------

``multi_gpu=True`` needs a running Dask-CUDA cluster, whose client is passed to ``GPU_RDF2Vec``:

.. code-block:: python

   from dask_cuda import LocalCUDACluster
   from dask.distributed import Client
   from rdf2vecgpu import GPU_RDF2Vec

   cluster = LocalCUDACluster()
   client = Client(cluster)
   model = GPU_RDF2Vec(multi_gpu=True, client=client)

On machines whose GPUs are connected only through PCIe (no NVLink), NCCL peer-to-peer transfers
can hang. Set ``NCCL_P2P_DISABLE=1`` in the environment there; do not set it on NVLink machines,
where it only slows down communication.

Verify your installation
------------------------

Run a quick import check:

.. code-block:: python

   import rdf2vecgpu
   import torch

   print("rdf2vecgpu version:", rdf2vecgpu.__version__)
   print("CUDA available:", torch.cuda.is_available())

Troubleshooting
---------------

- No matching distribution found / pip cannot find a wheel:

  - Upgrade pip: ``python -m pip install -U pip``
  - Check that you use Python 3.12 or newer on Linux (or WSL2).

- CUDA or GPU not detected:

  - Verify your PyTorch install: ``python -c "import torch; print(torch.cuda.is_available())"``
  - Install a CUDA-enabled PyTorch build matching your driver.

- Still stuck?

  - Check the package page on PyPI: https://pypi.org/project/rdf2vecgpu/
  - Open an issue on the `GitHub issue page <https://github.com/MartinBoeckling/rdf2vecgpu/issues>`__
