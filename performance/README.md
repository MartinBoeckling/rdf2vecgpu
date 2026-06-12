# Performance comparison setup
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

## Backend comparison: gensim vs single-GPU vs multi-GPU vs torchrec

[`bench_backends.py`](bench_backends.py) runs the **current** `RDF2VecConfig`
API across the four training backends on the same graph + hyperparameters and
records wall-clock, throughput (walks/sec), and **peak GPU memory** to
`performance/results/bench_results.jsonl` (one JSON row per run).

### Read this before interpreting numbers

torchrec's purpose is to **shard embedding tables that exceed a single GPU's
VRAM**. wikidata-5m (~4.8 M entities) at `vector_size=100` needs only ~3.8 GB
for both embedding tables — it **fits on one A6000**. On a table that fits,
torchrec's row-wise sharding + per-step all-to-all is pure overhead and is
expected to be **slower** than single-GPU or DDP. So treat this as **two
scenarios**:

1. **Fits-in-VRAM (e.g. wikidata-5m)** — all four backends run. Expectation:
   GPU backends ≫ gensim on throughput; torchrec competitive but with sharding
   overhead. Peak VRAM shows torchrec's per-GPU footprint is `~table/world_size`.
2. **Doesn't-fit (very large vocab)** — single-GPU and DDP **OOM**; only
   torchrec completes. This is torchrec's actual selling point. Use a large
   synthetic graph (or full Wikidata) whose `2 × vocab × dim × 4 B` exceeds one
   GPU's memory.

Always report **peak VRAM next to time** — the headline is the throughput ×
memory trade-off, not raw speed.

### Run it

```bash
DATA=data/wikidata5m/wikidata5m_kg.parquet

# CPU baseline (needs the gensim extra)
python performance/bench_backends.py --backend gensim     --data $DATA --tag wikidata5m

# single GPU
python performance/bench_backends.py --backend pytorch    --data $DATA --tag wikidata5m

# multi-GPU data-parallel (Lightning DDP over a Dask-CUDA cluster)
python performance/bench_backends.py --backend pytorch-mg --data $DATA --tag wikidata5m --nproc 2

# sharded embeddings (torchrun; NCCL_P2P_DISABLE=1 on PCIe boxes without NVLink)
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 \
  performance/bench_backends.py --backend torchrec --data $DATA --tag wikidata5m
```

Hyperparameters (`--walk-depth`, `--walk-number`, `--vector-size`, `--epochs`,
`--learning-rate`, …) default to the values in `evaluation_parameters.py`; hold
them constant across backends for a fair comparison.

### Results (fill in after running)

| backend     | launch              | walks/sec | train time (s) | peak VRAM (GiB/GPU) |
| ----------- | ------------------- | --------- | -------------- | ------------------- |
| gensim      | CPU                 |           |                | —                   |
| pytorch     | 1×GPU               |           |                |                     |
| pytorch-mg  | 2×GPU DDP           |           |                |                     |
| torchrec    | 2×GPU sharded       |           |                |                     |

> Raw rows accumulate in `performance/results/bench_results.jsonl`
> (`t_load_s`, `t_fit_transform_s` / `t_prepare_corpus_s` + `t_fit_from_walks_s`,
> `peak_vram_gib`, `world_size`, …). For torchrec the prep and training phases
> are timed separately because they run in different processes.