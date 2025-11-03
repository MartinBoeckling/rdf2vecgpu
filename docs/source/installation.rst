.. _installation_guide:

Installation Guide
==================

Overview
--------

``rdf2vecgpu`` is distributed on PyPI. Install with pip for the quickest setup. GPU acceleration is optional and depends on your system’s PyTorch/CUDA setup.

Prerequisites
-------------

- Python 3.12 or newer
- Linux, or Windows
- NVIDIA GPU with a compatible CUDA driver 12.x (Linux/Windows)

Note for macOS users:
  CUDA-based GPU acceleration is not available on macOS in most cases. The package will not be able to run.

Quick install (PyPI)
--------------------

Create and activate a virtual environment (recommended), then install:

.. code-block:: bash

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # Upgrade packaging tools
   python -m pip install -U pip setuptools wheel

   # Install rdf2vecgpu from PyPI
   python -m pip install -U rdf2vecgpu

GPU acceleration
---------------------------

To use the package with GPU acceleration, install the relevant CUDA 12.x drivers, then install ``rdf2vecgpu``:

1) Install PyTorch following the official selector for your OS/CUDA:
   https://pytorch.org/get-started/locally/

2) Verify PyTorch CUDA:

.. code-block:: python

   import torch
   print("CUDA available:", torch.cuda.is_available())

3) Install rdf2vecgpu (if not already installed):

.. code-block:: bash

   python -m pip install -U rdf2vecgpu

Notes:
- On Linux/Windows, ensure your NVIDIA driver and CUDA runtime match the PyTorch build.

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

   python -m pip install -U pip setuptools wheel
   python -m pip install -e .

   # Optional: install test/dev tools if provided by the project
   # python -m pip install -e ".[dev]"

Verify your installation
------------------------

Run a quick import check:

.. code-block:: python

   try:
       import rdf2vecgpu
       try:
           import torch
           cuda = torch.cuda.is_available()
       except Exception:
           cuda = False
       print("rdf2vecgpu imported OK")
       print("CUDA available:", cuda)
       v = getattr(rdf2vecgpu, "__version__", "unknown")
       print("rdf2vecgpu version:", v)
   except Exception as e:
       print("Import failed:", e)
       raise

Troubleshooting
---------------

- No matching distribution found / pip cannot find a wheel:
  - Upgrade pip: ``python -m pip install -U pip``
  - Ensure your Python version is supported (Python 3.8+).
  - On some platforms/architectures, building from source may require build tools.

- CUDA or GPU not detected:
  - Verify your PyTorch install: ``python -c "import torch; print(torch.cuda.is_available())"``
  - Install a CUDA-enabled PyTorch build matching your driver/runtime.
  - macOS does not support CUDA; use CPU.

- Permission errors on Linux:
  - Use a virtual environment, or add ``--user`` to pip installs.

- Still stuck?
  - Check the package page on PyPI: https://pypi.org/project/rdf2vecgpu/
  - Consult your CUDA/PyTorch installation
  - Open an issue on `Github issue page <https://github.com/MartinBoeckling/rdf2vecgpu/issues>`__
