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

## Data

- Synthetic graphs: download them from [Zenodo](https://doi.org/10.5281/zenodo.15368485)
  (`<model>_graph_<size>.parquet` for Barabási–Albert, Erdős–Rényi and growing random graphs of
  100, 1,000 and 10,000 vertices).
- FB15k-237: the train, validation and test splits merged into one graph, `fb15k_kg.parquet`.

Each graph also needs an N-Triples copy (`.nt`) next to its parquet file, which jRDF2Vec and
pyRDF2Vec read.

## Environments

Each framework runs in its own Python environment, passed to `benchmark.py`:

- rdf2vecgpu: the release wheel with the `gensim` extra (`pip install "rdf2vecgpu[gensim]"`) in a
  RAPIDS environment (see the main README).
- jRDF2Vec: the `jrdf2vec-1.3-SNAPSHOT.jar` and a Java runtime; `--jrdf2vec-python` is the Python
  that jRDF2Vec starts its gensim server with (Python 3.12, gensim 4.3.3, Flask 2.2).
- pyRDF2Vec: `pip install pyrdf2vec==0.2.3` (Python 3.10).

## ISWC 2025 setup

The scripts and pinned environments used for the ISWC 2025 paper are no longer maintained here.
They are in the [`v0.5.0` tag](https://github.com/MartinBoeckling/rdf2vecgpu/tree/v0.5.0/performance).
