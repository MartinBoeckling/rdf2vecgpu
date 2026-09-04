# Helix runbook — torchrec sharded Word2Vec (smoke → full Wikidata)

Everything here runs on **Helix** (bwForCluster, Heidelberg) except the rsync
commands, which run on a **dws login node**. Hardware/SLURM references:
<https://wiki.bwhpc.de/e/Helix/Hardware>, <https://wiki.bwhpc.de/e/Helix/Slurm>.

Why Helix (2026-09-03 assessment):
* **gpu8 nodes make full Wikidata a single-node job.** 8× A100/80GB = 640 GB
  VRAM (d100 tables 314 GB → ~39 GB/GPU) or 8× H200/141GB = 1128 GB (d200
  tables 629 GB → ~79 GB/GPU). All-to-all stays on NVLink/NVSwitch; the
  `gpu-single` partition allows 120 h.
* **gpu-multi has real InfiniBand** (gpu4-A100 nodes, 2× HDR200, ≤8 nodes /
  48 h) — the honest venue for the multi-node scaling experiment; dws
  inter-node is 10GbE TCP.

## 1. Workspace + transfer (dws side)

```bash
# on Helix once:
ws_allocate rdf2vec 60          # 60-day workspace; note the returned path
# on dws login node (2FA prompts):
WS=<workspace path printed above>
rsync -avP /ceph/mboeckli/rdf2vecgpu \
      mboeckli@helix.bwservices.uni-heidelberg.de:$WS/          # repo (~small)
rsync -avP /ceph/mboeckli/rdf2vec_bench/corpus/dbpedia2016 \
      mboeckli@helix.bwservices.uni-heidelberg.de:$WS/corpus/   # smoke corpus, 583 MB
```

Full-Wikidata corpus (only once the smoke passes; 304 GB + vocab):

```bash
rsync -avP /ceph/mboeckli/wikidata_rdf2vec_resource/data/wikidata/walks_v100/ \
      mboeckli@helix...:$WS/corpus/wikidata_v100/walks/
rsync -avP /ceph/mboeckli/wikidata_rdf2vec_resource/data/wikidata/vocab_new/word2idx.parquet \
      mboeckli@helix...:$WS/word2idx.parquet
```

Schema check (done 2026-09-03): `walks_v100/*.parquet` already matches the
`fit_from_walks` walks contract exactly (`vertices: list<int64>`,
`predicates: list<double>`). Only the vocab needs converting:

```bash
python performance/helix/make_vocab_parquet.py \
    $WS/word2idx.parquet $WS/corpus/wikidata_v100/vocab.parquet
```

## 2. Environment (Helix side)

```bash
cd $WS/rdf2vecgpu
bash performance/helix/setup_env.sh          # venv at ~/venvs/rdf2vecgpu
```

Pins mirror the working dws env: torch 2.9.1 / torchrec 1.4.0 / fbgemm-gpu
1.4.1 (cu128), lightning 2.5.6, cudf+cugraph 25.6 (import-time deps only —
training never touches cuGraph).

## 3. Smoke (dbpedia2016, ~15–60 min)

```bash
cd $WS/rdf2vecgpu
WS=$WS sbatch performance/helix/train_torchrec_gpu8.sbatch   # 2x A100, 1 epoch
```

Pass = `results/gpu8/run.jsonl` written, loss decreasing in the log, topology
matrix shows NV* links. Optionally repeat with `--gres=gpu:A100:4`,
`GPUS=4` to watch throughput scale, and run the IB multi-node variant:

```bash
WS=$WS sbatch performance/helix/train_torchrec_multinode_ib.sbatch  # 2 nodes x 4 GPUs
```

## 4. Full-Wikidata run (after smoke + corpus transfer)

d100 on an A100 gpu8 node:

```bash
WS=$WS sbatch --gres=gpu:A100:8 --cpus-per-task=48 --mem=800G --time=96:00:00 \
  --export=ALL,WS=$WS,GPUS=8,CORPUS=$WS/corpus/wikidata_v100,OUT=$WS/results/wikidata_d100,VECTOR_SIZE=100,EPOCHS=5,LR=0.10 \
  performance/helix/train_torchrec_gpu8.sbatch
```

d200 → same but `--gres=gpu:H200:8`, `VECTOR_SIZE=200` (A100/80 is too tight:
79 GB/GPU of tables before activations).

Open items to watch on the first full-scale attempt:
* **Host RAM / loader**: 304 GB of walks streamed via the row-striped loader —
  watch RSS; gpu8 nodes have 2.2 TB so headroom is large.
* **Wall-time vs 120 h cap**: unknown until the first epoch's steps/s is in;
  the torchrec checkpointing config (`RDF2VecConfig`) supports resume if 5
  epochs don't fit one job.
* **Negative sampler**: `strategy="unigram"` needs token_counts — the bench
  path derives them from the walks; verify it doesn't try to materialise
  anything at 393M-vocab scale before launching 5 epochs (run EPOCHS=1 first).
* **Quality gate**: export KGrEAT TSV (`EXPORT_KGREAT=...`) on a d100 run and
  score with the existing GEval harness on dws for the §4-band comparison.
