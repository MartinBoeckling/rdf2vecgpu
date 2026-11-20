.. _performance:

Performance
===========

.. raw:: html

   <link rel="stylesheet" href="_static/benchmarks/charts.css">
   <script src="_static/benchmarks/charts.js" defer></script>

rdf2vecgpu 0.6.0 compared with the two most widely used RDF2Vec implementations,
`jRDF2Vec <https://github.com/dwslab/jRDF2Vec>`__ (Java, CPU) and
`pyRDF2Vec <https://github.com/IBCNServices/pyRDF2Vec>`__ (Python, CPU), on the graphs of the
`ISWC 2025 paper <https://doi.org/10.1007/978-3-032-09530-5_14>`__.

On FB15k-237, rdf2vecgpu 0.6.0 trains on one GPU 2.9–3.6× faster than jRDF2Vec and 40–90× faster
than pyRDF2Vec, both with 16 CPU threads. With 100 walks of depth 4 it takes 43 s end to end,
against 130 s for jRDF2Vec, 29 min for pyRDF2Vec and 9.9 min for rdf2vecgpu 0.5.0 on the same
machine. On graphs with a few hundred vertices the start-up of the GPU libraries (about 15 s)
dominates, and the CPU frameworks are as fast or faster.

Setup
-----

- **What is measured**: wall time of one process, end to end: reading the graph, generating the
  walks, training and writing the vectors. Each framework runs in its own environment, one run
  at a time; the time shown is the median of 3 runs.
- **Hardware**: one node with one NVIDIA RTX A6000 (48 GB). The CPU frameworks (jRDF2Vec,
  pyRDF2Vec and rdf2vecgpu's gensim backend) get 16 CPU threads.
- **Training**: random walks, skip-gram, 5 epochs, 100 dimensions, window 5, 5 negative samples,
  ``min_count=1``; every other setting at each framework's default. rdf2vecgpu runs with
  ``backend="pytorch"`` (GPU training) and ``backend="gensim"`` (GPU walks, CPU training).
- **Limits**: a run is stopped after 12 hours. A framework that times out or runs out of memory is
  not started on larger runs of the same graph.
- **Memory**: peak GPU memory as reported by ``nvidia-smi`` (what PyTorch and RAPIDS have reserved,
  which can be more than they use at any moment) and peak host memory (PSS) of all the run's
  processes, both sampled once per second.

Speed-up
--------

How many times faster rdf2vecgpu (PyTorch) is than jRDF2Vec and pyRDF2Vec on the same run: the
reference's wall time divided by rdf2vecgpu's. Points right of the 1× line are runs where
rdf2vecgpu is faster. A hollow marker is a lower bound: the reference hit the 12-hour limit.

.. raw:: html

   <div class="bench-chart" data-chart="speedup" data-base="_static/benchmarks/">
     <img class="only-light" src="_static/benchmarks/speedup-light.svg"
          alt="Speed-up of rdf2vecgpu over jRDF2Vec and pyRDF2Vec per run, log scale; the values are in the tables below.">
     <img class="only-dark" src="_static/benchmarks/speedup-dark.svg"
          alt="Speed-up of rdf2vecgpu over jRDF2Vec and pyRDF2Vec per run, log scale; the values are in the tables below.">
   </div>

FB15k-237
---------

FB15k-237 (14,541 entities, 237 relations, 310,116 triples) with 100, 500 and 1000 walks per vertex of depth 4, 8
and 12. Hover over a point for its median and range; click a legend entry to highlight one
framework.

.. raw:: html

   <div class="bench-chart" data-chart="fb15k" data-base="_static/benchmarks/">
     <img class="only-light" src="_static/benchmarks/fb15k-light.svg"
          alt="Wall time on FB15k-237 by walk depth for 100, 500 and 1000 walks, log scale; the values are in the table below.">
     <img class="only-dark" src="_static/benchmarks/fb15k-dark.svg"
          alt="Wall time on FB15k-237 by walk depth for 100, 500 and 1000 walks, log scale; the values are in the table below.">
   </div>

Hollow markers are runs that hit the 12-hour limit. The time axis is logarithmic: each gridline is
ten times the one below.

Synthetic graphs
----------------

Barabási–Albert, Erdős–Rényi and growing random graphs of 100, 1,000 and 10,000 vertices, 100 walks
of depth 4. On the smallest graphs every framework takes seconds and the start-up of the GPU
libraries dominates rdf2vecgpu's time; there pyRDF2Vec is the fastest.

.. raw:: html

   <div class="bench-chart" data-chart="synthetic" data-base="_static/benchmarks/">
     <img class="only-light" src="_static/benchmarks/synthetic-light.svg"
          alt="Wall time on synthetic graphs by number of vertices, log scale; the values are in the table below.">
     <img class="only-dark" src="_static/benchmarks/synthetic-dark.svg"
          alt="Wall time on synthetic graphs by number of vertices, log scale; the values are in the table below.">
   </div>

Tables
------

The speed-up columns divide the reference framework's time by rdf2vecgpu's PyTorch time. "> 12.0 h"
marks a run that hit the time limit, "–" a run that was not started because a smaller run of the
same framework had already failed. The memory tables give the highest peak over the 3 runs, in GB.
Some pyRDF2Vec runs were repeated twice instead of three times, and their time is the median of
2 runs: on FB15k-237 100 and 500 walks of depth 8 and 12 and 1000 walks of depth 4 and 8, and the
Erdős–Rényi graph of 1,000 vertices.

.. include:: _generated/benchmark_tables.inc

Embedding quality
-----------------

Faster training only helps if the embeddings stay as good. rdf2vecgpu's PyTorch models train the
way word2vec does (summed loss, SGD with a linearly decaying learning rate, subsampling of frequent
tokens), so they are compared here with the implementations RDF2Vec usually runs on: gensim for
skip-gram and CBOW, and wang2vec's structured skip-gram for order-aware skip-gram (RDF2Vec\ :sub:`oa`).

- **Benchmark**: the data-mining tasks of `KGrEAT <https://github.com/nheist/KGrEAT>`__ on DBpedia
  2016 (AAUP, Cities, Forbes, Metacritic albums and movies; classification and regression). Each
  score is the best of KGrEAT's estimators for the task, on the task's entities found in the graph.
- **Walks**: in the style of RDF2Vec Light, 500 random walks of depth 4 from each task entity over
  the part of DBpedia within 4 hops of them. Every model trains on the same walks.
- **Training**: 100 dimensions, window 5, 5 epochs, 5 negative samples, ``sample=0.001``, each
  learning rate 0.025 for every model and, for rdf2vecgpu, the default ``batch_size``. Three
  training seeds per model.

Skip-gram matches or beats gensim on every task, by the widest margin on the Cities regression.
CBOW is ahead of gensim on most tasks, most clearly the Metacritic albums classification, but
behind on the AAUP and Cities regressions. Order-aware skip-gram is within 0.02 of wang2vec on
every task, ahead on some and behind on others.

.. list-table:: KGrEAT on DBpedia 2016 (RDF2Vec Light), mean ± standard deviation over 3 seeds
   :header-rows: 1
   :widths: auto

   * - Task
     - rdf2vecgpu skip-gram
     - gensim skip-gram
     - rdf2vecgpu CBOW
     - gensim CBOW
     - rdf2vecgpu order-aware
     - wang2vec structured skip-gram
   * - AAUP, accuracy ↑
     - 0.576 ± 0.004
     - 0.573 ± 0.007
     - 0.562 ± 0.009
     - 0.556 ± 0.005
     - 0.560 ± 0.011
     - 0.554 ± 0.001
   * - AAUP, RMSE ↓
     - 0.563 ± 0.002
     - 0.567 ± 0.002
     - 0.597 ± 0.011
     - 0.572 ± 0.010
     - 0.569 ± 0.013
     - 0.589 ± 0.010
   * - Cities, accuracy ↑
     - 0.733 ± 0.020
     - 0.733 ± 0.003
     - 0.660 ± 0.012
     - 0.653 ± 0.020
     - 0.693 ± 0.003
     - 0.694 ± 0.010
   * - Cities, RMSE ↓
     - 0.502 ± 0.020
     - 0.597 ± 0.020
     - 0.601 ± 0.009
     - 0.559 ± 0.011
     - 0.580 ± 0.020
     - 0.592 ± 0.008
   * - Forbes, accuracy ↑
     - 0.583 ± 0.007
     - 0.573 ± 0.006
     - 0.576 ± 0.005
     - 0.550 ± 0.010
     - 0.582 ± 0.006
     - 0.585 ± 0.005
   * - Forbes, RMSE ↓
     - 0.589 ± 0.007
     - 0.587 ± 0.001
     - 0.585 ± 0.004
     - 0.591 ± 0.009
     - 0.594 ± 0.005
     - 0.594 ± 0.005
   * - Metacritic albums, accuracy ↑
     - 0.750 ± 0.006
     - 0.740 ± 0.002
     - 0.723 ± 0.012
     - 0.624 ± 0.010
     - 0.730 ± 0.003
     - 0.743 ± 0.001
   * - Metacritic albums, RMSE ↓
     - 0.445 ± 0.002
     - 0.453 ± 0.001
     - 0.472 ± 0.005
     - 0.482 ± 0.009
     - 0.455 ± 0.002
     - 0.454 ± 0.001
   * - Metacritic movies, accuracy ↑
     - 0.727 ± 0.009
     - 0.702 ± 0.002
     - 0.707 ± 0.013
     - 0.700 ± 0.003
     - 0.699 ± 0.009
     - 0.715 ± 0.005
   * - Metacritic movies, RMSE ↓
     - 0.458 ± 0.003
     - 0.466 ± 0.005
     - 0.476 ± 0.005
     - 0.480 ± 0.004
     - 0.453 ± 0.001
     - 0.455 ± 0.004

Reproducing
-----------

The benchmark harness is in the `performance folder
<https://github.com/MartinBoeckling/rdf2vecgpu/blob/main/performance>`__. The charts and tables on
this page are generated from its ``runs.jsonl`` output with
``performance/benchmark_charts.py``, which refuses to combine runs from different nodes.
