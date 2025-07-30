gpuRDF2Vec
==========

A scalable GPU-based implementation of RDF2Vec embeddings for large and dense Knowledge Graphs.

.. image:: https://github.com/MartinBoeckling/rdf2vecgpu/blob/main/img/github_repo_header.png
   :alt: RDF2VecGPU Image
   :align: center
   :width: 600px

.. note::
   Licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Sections
   
Repository Setup
----------------

The repository builds on two major libraries: **PyTorch Lightning** and **RAPIDS (cuDF, cuGraph)**. For CUDA 12.6, install as follows:

.. code-block:: bash

   pip install torch torchvision torchaudio

.. code-block:: bash

   pip install \
       --extra-index-url=https://pypi.nvidia.com \
       "cudf-cu12==25.4.*" "dask-cudf-cu12==25.4.*" \
       "cugraph-cu12==25.4.*" "nx-cugraph-cu12==25.4.*"

Environment setup files:

- `Conda environment <../performance/env_files/rdf2vecgpu_environment.yml>`_
- `Requirements file <../performance/env_files/rdf2vecgpu_requirements.txt>`_

Overview
--------

**gpuRDF2Vec** provides a GPU-accelerated implementation of RDF2Vec for high-performance graph embedding generation.

Capabilities include:

- Fast walk generation using cuGraph
- Batched Word2Vec training using PyTorch Lightning
- Pluggable RDF data loaders (CSV, TXT, Parquet, NT, RDFlib formats)
- Industrial-scale scalability and reproducibility

Quick Start
-----------

.. code-block:: python

   from src.gpu_rdf2vec import GPU_RDF2Vec

   model = GPU_RDF2Vec(
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

   edge_data = model.load_data("data/wikidata5m/wikidata5m_kg.parquet")
   embeddings = model.fit_transform(edge_df=edge_data, walk_vertices=None)
   embeddings.to_parquet("data/wikidata5m/wikidata5m_embeddings.parquet", index=False)

Supported input formats:

- CSV, TXT, Parquet, NT
- All `RDFlib-supported formats <https://rdflib.readthedocs.io/en/stable/plugin_parsers.html>`_

Implementation Details
-----------------------

Key engineering improvements over CPU RDF2Vec:

1. **GPU-native Walk Extraction**:
   - Fully GPU-side random walks and BFS via cuGraph
   - Massively parallel node replication for walk creation

2. **cuDF→PyTorch Handoff**:
   - cuDF-backed DataLoader
   - DLPack tensor conversions eliminate CPU bottlenecks

3. **Optimized Word2Vec**:
   - Auto-batch sizing based on GPU memory
   - Kernel fusion and C++ backend processing

4. **Distributed Training**:
   - Multi-GPU via PyTorch Distributed and NCCL
   - `all_reduce` for synchronized gradient sharing

Roadmap
-------

- [ ] Order-aware Word2Vec [Ling et al. (2015)](https://aclanthology.org/N15-1142.pdf)
- [ ] Spilling to single-GPU for memory-bound training
- [ ] Weighted walks for spatial datasets
- [ ] Integration with `wandb` and `mlflow` for experiment tracking

Report Issues and Bugs
----------------------

Please open an issue with the label **Bug** and provide:

- **Environment**: OS, Python, CUDA, PyTorch, cuDF versions
- **Reproduction steps**: Code or CLI input
- **Dataset**: Format & size
- **Observed behavior** vs **expected behavior**
- **Error logs** or stack traces

We aim to respond within **3 business days**. For fixes, open a PR referencing the issue.

License
-------

This project is licensed under the MIT License.
