# Performance benchmarks

## End-to-end benchmark (docs performance page)

`benchmark.py` times rdf2vecgpu, jRDF2Vec and pyRDF2Vec end to end (reading the graph, walks,
training, writing the vectors) on FB15k-237 and the synthetic graphs, one run at a time, each in a
fresh process in the framework's own environment. `benchmark_runner.py` is the per-run script
that `benchmark.py` starts in each environment. Every run appends one JSON line to
`<output>/runs.jsonl`; re-running with the same output directory resumes.

```bash
python performance/benchmark.py --data data/benchmark --output results/0.6.0 \
    --wheel-python /path/to/rdf2vecgpu-env/bin/python \
    --jrdf2vec-python /path/to/jrdf2vec-env/bin/python --jar jrdf2vec-1.3-SNAPSHOT.jar \
    --pyrdf2vec-python /path/to/pyrdf2vec-env/bin/python \
    --frameworks gpu-random-pytorch gpu-random-gensim jrdf2vec pyrdf2vec --timeout 43200
```

`--data` holds `fb15k_kg.{parquet,nt}` and `<model>_graph_<size>.{parquet,nt}` for the synthetic
graphs (see below). On Slurm, `--write-plan` lists the runs and `--only ID` runs one, so each run
can be its own job. Pin the CPU frameworks to the same number of cores (e.g. `--cpus-per-task=16`):
they use every core the process may use.

`benchmark_charts.py` turns `runs.jsonl` files into the charts (Vega-Lite specs and SVGs) and
tables of the [performance page](https://rdf2vecgpu.readthedocs.io/en/latest/performance.html):

```bash
pip install vl-convert-python
python performance/benchmark_charts.py results/0.5.0/runs.jsonl results/0.6.0/runs.jsonl \
    --out docs/source/_static/benchmarks --table docs/source/_generated/benchmark_tables.inc \
    --subtitle "rdf2vecgpu 0.6.0 · RTX A6000, 16 CPU threads · median of 3"
```

When several files hold the same run, the later file wins, so the reference frameworks can come
from an earlier measurement on the same machine. It refuses to combine runs from different
machines, as wall times are not comparable.

## ISWC 2025 setup

This folder contains everything you need to build and run the performance benchmarks for the GPU-accelerated RDF2Vec implementation. Use these benchmarks to measure throughput, memory usage, and speedup over the CPU baseline.

## Dataset creation
Please download the set of synthetic datasets from the following [Zenodo repository](https://doi.org/10.5281/zenodo.15368485) and place them under `data/generated_graphs`. Please use the FB15k-237 repository as well as the wikidata-5m repository in order to download the data assets. For both the FB15k-237 dataset as well as the [merge_text_file script](data_preparation/merge_text_file.py) to merge all the different splits together. 

For both the jrdf2vec library as well as the sparkkgml library please convert the parquet files into a ttl file.

Scripts:
- To calculate the graph statistics please run the following [script](graph_statistics.py)
- To run the jrdf2vec performance comparison, please use the following [script](jrdf2vec_based_performance.py)
- To run the pyrdf2vec performance comparison, please use the following [script](pyrdf2vec_based_performance.py)
- To run the sparkrdf2vec performance comparison, please use the following [script](spark_rdf2vec_performance.py)
- To run the gpu_rdf2vec performance comparison, please use the following [script](gpu_rdf2vec_performance.py)