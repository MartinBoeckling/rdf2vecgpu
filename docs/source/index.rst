gpuRDF2vec documentation
==========

gpuRDF2Vec is a scalable GPU-based implementation of RDF2Vec embeddings for large and dense Knowledge Graphs.

.. image:: _static/img/rdf2vecgpu_logo.png
   :alt: RDF2VecGPU Image
   :align: center
   :width: 600px

.. note::
   Licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.

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


Report Issues and Bugs
----------------------

Please open an issue with the label **Bug** and provide using the following template under the `Github issue page <https://github.com/MartinBoeckling/rdf2vecgpu/issues>`__:

- **Environment**: OS, Python, CUDA, PyTorch, cuDF versions
- **Reproduction steps**: Code or CLI input
- **Dataset**: Format & size
- **Observed behavior** vs **expected behavior**
- **Error logs** or stack traces

We aim to respond within **3 business days**. For fixes, open a PR referencing the issue.

License
-------

This project is licensed under the MIT License.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Installation

   installation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: gpuRDF2vec Usage

   rdf2vecgpuusage
   gettingstarted
   dataload
   training
   vectorretrieval


Indices and tables
------------------

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
