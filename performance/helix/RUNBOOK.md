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

GPU unit tests need a GPU allocation (fbgemm imports libcuda at import time);
run them as a gate and chain the real runs behind it — dependents are auto-
cancelled if the gate fails (`kill_invalid_depend`):

```bash
T=$(sbatch --parsable performance/helix/gpu_tests.sbatch)   # 2x A100, ~5 min, exit = pytest status
sbatch --dependency=afterok:$T ... performance/helix/train_torchrec_gpu8.sbatch
```

## 4. Full-Wikidata run (after smoke + corpus transfer)

**Scale, measured 2026-09-04:** `wikidata_v100/walks` = 11,004 shards,
**39.29 billion walks** (393M start vertices × 100 walks; depth 4, 34% of
vertex slots are `-1` sentinels), vocab 392,910,288. Two consequences that
the dbpedia smoke cannot show you:

* **Precompute token counts once** (`token_counts.sbatch`, CPU, 48 cores).
  Without the cache every torchrun rank streams the whole 311 GB corpus
  before step 0 — measured 3.4 s/shard → ~10 h per rank with the GPUs idle.
* **Use a large `PAIR_BATCH_SIZE`.** It is split *across* ranks by row
  stride, so 131072 on 8 GPUs is only 16k pairs/rank/step and 18.0M
  steps/epoch (~92 h at the smoke's 54 steps/s). `1048576` → 131k
  pairs/rank, 2.25M steps/epoch, ~12 h/epoch at the same step rate. The
  loss is sum-reduced under row-wise Adagrad, so lr stays per-pair.

Checkpointing is on by default (`$OUT/ckpt`, every `CHECKPOINT_INTERVAL_MIN`
= 120 min + epoch end, auto-resume on resubmit). One d100 checkpoint is
~300 GB written shard-parallel; `--checkpoint-keep-last 2`.

**Host memory (`--mem`) is not free to shrink.** Measured 2026-09-04 on the
dbpedia smoke: 2 ranks idle at **27 GB RSS** (cudf/RMM/torch + pyarrow read
chunks) and DCP stages each rank's shard through host RAM on save — a 32 GB
job was OOM-killed at the epoch-end checkpoint after the mid-run one had
succeeded. Budget ≈ 13 GB/rank baseline + the rank's table shard for the
checkpoint; the gpu8 nodes have 2 TB, so submit the full run with `--mem=1500G`
(accepted by sbatch; to raise an already *pending* job use the MB form,
`scontrol update JobId=<id> MinMemoryNode=1500000` — `MinMemoryNode=1500G` is
rejected). The same smoke re-run at 64 GB peaked at **38 GB MaxRSS** for 2 ranks
with both checkpoints (mid-run + epoch end, 9.4 GB each) written.

**The first full-Wikidata attempt (2026-09-06, 8× H200, job 14593196) was
OOM-killed at 1.5 TB host RAM before step 0.** Cause: the model was built on
the default device — every rank allocated the full 2 × 393M × 100 fp32 tables
(314 GB) in host RAM before `.to(gpu)`, 8 × 314 GB = 2.5 TB. The trainer now
builds the module on the meta device and DMP materialises only the local shard
on the GPU; training also no longer loads the 393M-row vocab frame (~20 GB of
GPU memory + a 7.4 GB read per rank) — `transform()` reads it lazily. Expected
host budget after the fix: ~15–20 GB/rank baseline + the 37 GB shard staged at
each checkpoint → well under 1 TB for 8 ranks.

Smoke result at `PAIR_BATCH_SIZE=1048576` (2× A100-40GB, d200): **13.6–14.3
steps/s = 13.7M pairs/s vs 7.1M at 131072** (1.94×); loss at step 0 scales by
exactly the batch ratio (sum loss), and falls 800k → 30k over the epoch.

End-to-end dbpedia2016 **d100, 1 epoch + KGrEAT export** (2026-09-05, 2× A100-40GB,
`PAIR_BATCH_SIZE=1048576`, row-wise plan): fit **135 s** (26 steps/s, loss 803k →
35k), export of all 6,306,612 entities = 6.2 GB TSV in ~3.5 min, job MaxRSS 36 GB
(`--mem=96G`), peak VRAM 9.0 GiB/GPU. Results in `results/dbpedia_d100_e1/`.
Re-run 2026-09-06 after the meta-device + lazy-vocab fix (`dbpedia_d100_e1_v2`):
fit **103 s**, peak VRAM **4.2 GiB/GPU** (the vocab frame alone was ~4.7 GiB on
dbpedia — ~20 GB/GPU at Wikidata scale), MaxRSS 30.5 GB, same 6,306,612-row TSV.

```bash
# once per corpus (~minutes on 48 cores):
sbatch --export=ALL,WS=$WS,CORPUS=$WS/corpus/wikidata_v100 performance/helix/token_counts.sbatch
# d100, 1 epoch first — gives the real steps/s before committing to 5:
WS=$WS sbatch --gres=gpu:A100:8 --cpus-per-task=48 --mem=1500G --time=48:00:00 \
  --export=ALL,WS=$WS,GPUS=8,CORPUS=$WS/corpus/wikidata_v100,OUT=$WS/results/wikidata_d100_e1,VECTOR_SIZE=100,EPOCHS=1,LR=0.10,PAIR_BATCH_SIZE=1048576,RACE_GROUP=wd_d100_e1 \
  performance/helix/train_torchrec_gpu8.sbatch
```

`--gres=gpu:H200:8` is the same job on the H200 gpu8 node (141 GB/GPU, ~2.4×
the HBM bandwidth of A100; the measured Wikidata epoch used ~1.3% of it, see
the throughput notes below, so the H200 advantage is smaller than that ratio).
Queue both with the same `RACE_GROUP` — sibling job names MUST be
`<RACE_GROUP>_<suffix>`; a job that merely starts with the group string is not
a sibling (2026-09-08: `dbpx_prof2` was cancelled by group `dbpx`) — and the
script keeps whichever gets *past start-up* first: a job claims the race
(atomic winner file under `$WS/results`) only after its first progress line, then
cancels the pending sibling; a start-up failure leaves the sibling queued (validated
2026-09-06 with two dbpedia siblings: claim logged right after `step 0`, sibling
cancelled one minute after the winner started) — no
external watcher. Helix has no B200. Chain the pair behind a validation job
with `--dependency=afterok:<id>` so they release the moment it passes.

**Full-Wikidata d100, 1 epoch (2026-09-08, job 14636218, 8× H200 `o09c02` —
won the race, the A100 sibling was cancelled while still pending): 671,187
steps at 79.8 steps/s (1.05M pairs/step), fit 9,353 s = 2.6 h, loss 551k →
~80k, peak VRAM 28.5 GiB/GPU, job MaxRSS 1.12 TB (`--mem=1500G`), wall
2 h 38 min. Two kept checkpoints in `results/wikidata_d100_e1_h200/ckpt`
(293 GB each). So 5 epochs ≈ 13 h on the H200 node (A100-80GB: expect
1.5–2×, the step is bandwidth-bound); the queue wait was 37 h both times.**

Leave `EXPORT_KGREAT` unset on a long training run and export afterwards with
an **export-only resubmit**: the same submit line plus
`EXPORT_KGREAT=$OUT/entities.tsv,EXPORT_ENTITY_REGEX=^http://www.wikidata.org/entity/Q`
(same `OUT`, so `CHECKPOINT_DIR` resolves to the finished run's `ckpt`). It
resumes at the epoch boundary, trains nothing, prints `[torchrec] resumed …`
(what the race claim keys on in that case) and streams the filtered table to
the TSV. The exporter (`bench_backends.py --export-kgreat`) streams each rank's
row block through `GPU_RDF2Vec.iter_local_embeddings` in 1M-row chunks: no
device-side vocab join (the `transform()` path needs ~110 GB/GPU at 393M
tokens — more than an A100-80GB) and no whole-shard pandas frame; the vocab
slice comes from the row groups of `vocab.parquet` that cover the block.
`--export-entity-regex` (RE2) and `--export-entity-list <file>` (one IRI per
line, e.g. the GEval gold entities) filter rows before writing — unfiltered,
the Wikidata TSV would be ~320 GB.

First full-scale export attempt (2026-09-14, job 14697728, 8× H200): rank 0
died in `_read_vocab_slice` with `ArrowInvalid: offset overflow while
concatenating arrays` two minutes in. A rank block is 49,113,786 IRIs =
2.6 GiB of entity bytes — over the 2 GiB cap of a pyarrow `string` array —
and the Wikidata `vocab.parquet` row groups are *not* in token-id order
(dbpedia2016's are), so the reader's sort materialised the whole block as one
`string` array. dbpedia2016 never hit it: 0.04 GiB per rank. The reader now
promotes the column to `large_string` on read (verified on the real rank-0
block: 10 s, 10 GB RSS); `RDF2VEC_BIG_TESTS=1 pytest -k over_2gib` runs the
2.3 GB regression. The e5 sibling (14697730) failed differently the same
night: three of eight ranks never produced their first batch inside the NCCL
watchdog's 10 min (nothing in the log from them), so `bench_backends.py` now
opens the process group with a 30 min timeout
(`RDF2VEC_TORCHREC_PG_TIMEOUT_MIN`) + `device_id`, every rank logs
`first batch of epoch N ready after S s`, and a rank still waiting after
5 min dumps its thread stacks (`[torchrec] STALL …`). The A100 siblings were
still pending, so the two H200 jobs were resubmitted with the same
`RACE_GROUP`s behind a fresh `gpu_tests.sbatch` gate.

d200 → same but `--gres=gpu:H200:8`, `VECTOR_SIZE=200` (A100/80 is too tight:
79 GB/GPU of tables before activations).

Open items to watch on the first full-scale attempt:
* **Host RAM / loader**: 304 GB of walks streamed via the row-striped loader —
  watch RSS; gpu8 nodes have 2.2 TB so headroom is large.
* **Wall-time vs 120 h cap**: unknown until the first epoch's steps/s is in;
  the torchrec checkpointing config (`RDF2VecConfig`) supports resume if 5
  epochs don't fit one job.
* **Sharding plan**: resolved 2026-09-05 — left to its cost model the planner
  placed dbpedia2016's 6.3M-row tables *table-wise* (one whole table per GPU),
  and the KGrEAT export crashed on the rank without the in-table
  (`DistributedModelParallel has no attribute in_collection`). `_build_sharding_plan`
  now pins every table to row-wise, so each rank owns one contiguous row
  block — the layout extraction, `_local_row_range` and the DCP shard files
  assume. Checkpoints written before the fix carry the table-wise layout.
* **Negative sampler**: resolved 2026-09-04 — the sampler now builds one
  float64 CDF per run and draws with `searchsorted` (the previous per-step
  `pow` + `torch.multinomial` re-read the whole 393M-entry weight vector every
  step, and an fp32 cumsum at that length zeroes rare tokens' probability).
* **Loader** (spike/loader-overlap worktree, 2026-09-06): measured — cold GPFS
  shard reads are 0.03 s (not the problem), but walks are SHORT: Wikidata yields
  17 of the nominal 60 pairs/walk (40% single-vertex stubs), dbpedia2016 19, so
  the row-count batches carried <1/3 of `PAIR_BATCH_SIZE` and 3.5× the steps.
  Batches are now cut by cumulative pair count (`_PairBudgetBatcher`,
  rank-independent cut points so ranks stay in lockstep) and decoded in a
  prefetch thread (`_ThreadPrefetch`); `_PairsFromWalks` is module-level; stub
  walks (< 2 valid vertices, 0 pairs) are dropped at decode (`drop_stubs`, on by
  default in pair-budget mode). Same pairs and batch count; with shuffle on, the
  within-chunk permutation runs over the surviving rows, so batch composition is
  a different (equally valid) shuffle. Measured on 6 real Wikidata shards: an
  83%-stub region → 56% less loader CPU time, a 0%-stub region → no change;
  ~40% stubs corpus-wide → roughly a quarter to a third less loader CPU work.
  Measured on dbpedia2016 d100, 2× A100 (`results/dbpedia_d100_e1_loader` vs
  `_v2`): step-0 loss 2.18M vs 0.80M (= 1.05M vs 0.39M real pairs per global
  step), ~990 steps vs 3,079, 14.8 vs 30.6 steps/s = **7.8M vs 5.9M pairs/s/rank**,
  fit **69.6 s vs 103 s** (1.48×), peak VRAM 6.7 vs 4.2 GiB, same 6,306,612-row TSV.
  5 epochs (`dbpedia_d100_e5_loader` vs `_e5`): ~4,900 vs 15,400 steps, fit
  **427 s vs 605 s** (1.42×). Extrapolated to Wikidata 8-GPU: ~640k steps/epoch
  instead of 2.25M. Still
  open: every rank decodes all 311 GB and keeps 1/8 (cheap: 0.25 s/shard warm),
  and the per-step `all_reduce`+`.item()` join.
* **Where the Wikidata epoch's time went** (job 14636218 log + sacct): steady
  median 79.0 steps/s (p95 80.3) from step 200 on; 857 s (9%) lost to ten
  recurring windows at ~31 steps/s (~2,800 steps each, every ~50k steps —
  stub-dense corpus regions where the loader decodes ~6 walks per surviving
  walk), five 13–46 s all-rank stalls (one is the 2-hourly 293 GB checkpoint)
  and ~80 s start-up. 83M pairs/s aggregate = 10.4M/GPU × 7 rows × 400 B ≈
  60 GB/s per GPU, ~1.3% of H200 HBM bandwidth — so NOT bandwidth-bound; the
  per-rank pair rate is close to the 2× A100-40GB dbpedia runs (7.8M/s/rank),
  which points at per-step fixed costs (Python step loop, all-to-all latency,
  the join sync). CPU 11.9 of 48 cores busy on average (~1.5/rank), 2.52 TB
  read (8 × the 304 GB corpus), 600 GB written, MaxRSS 1.12 TB (cgroup figure,
  likely includes page cache). Vocab: 392.9M tokens = 124.9M `entity/Q` items,
  125M `Special:EntityData` document nodes, ~140M external URLs (DOIs,
  catalogues — leaf nodes, hence the 40% stub walks), 0.8% literals.
* **Step-time attribution:** `RDF2VEC_TORCHREC_PROFILE=1` (diagnostic only —
  synchronises the device every step) prints per-rank host ms/step for
  drain / next / join / negatives / forward / backward / log, CUDA-event GPU
  time for forward+backward, and a 20-step `torch.profiler` kernel/collective
  table on rank 0 (`[torchrec][profile] …` lines).
* **Profile result (2026-09-08, dbpedia d100 e1, 2× A100-40GB, 525k
  pairs/rank):** 68 ms/step; the GPU chain (CUDA events, fwd+bwd) was 65 ms
  of it, the host loop 16.6 ms (2/3 of that in `.item()` syncs). GPU time:
  33% `torch.bmm` negative logits (B tiny GEMMs), 14% NCCL all-to-all, 10%
  torchrec id permute, 8% `index_add_` (index_select backward), ~12% output
  dist, **3% fbgemm lookup + RowWiseAdagrad** — the tables are not the cost.
  The `bmm` was replaced by a fused multiply-reduce (loss-identical, in main
  since 2026-09-08 22:00, gate 14699380). Same-node comparison on `o02c02`:
  **11.7 → 15.5 steps/s (1.32×)** (jobs 14638968 vs 14697733), consistent
  with the profiled GPU chain dropping 65 → 42.6 ms. **Node identity is a
  confound of the same size:** GEMM builds ran 11.7 (`o02c02`) vs 14.5–14.8
  (`p08c01/p08c02`); the fused build 15.5 (`o02c02`) vs 21.8–22.1 (`o05c01`,
  fit 57–58 s, jobs 14698160/14698161). A per-step `torch.cuda.synchronize()`
  (`RDF2VEC_TORCHREC_STEP_SYNC=1`, default off) was A/B-tested on `o05c01`
  and does nothing (22.1 vs 21.8). Always note the host and A/B on one node.
* **Rank scaling (fused build, `o05c01`, profiled):** 2 GPUs 22.0 steps/s,
  GPU chain 42.6 ms at 525k pairs/rank; 4 GPUs **31.6 steps/s**, GPU chain
  29 ms at 262k pairs/rank (job 14697718) — halving the per-rank work cut the
  chain by 1.47×, so ~15 ms of it is rank-count-independent (all-to-all +
  permute latency: with the bmm gone the top CUDA rows are NCCL SendRecv
  7.6 ms/step and fbgemm permute 4.9 ms/step). The host loop is ~15 ms/step
  at any rank count (forward 8.4–8.7 ms of torchrec Python), so at 8 ranks
  × 131k pairs the step is host-bound: the Wikidata e5 run will show how much
  of the 12.5 ms/step the fused kernel recovers there. Next lever if more is
  needed: cut host work per step (larger PAIR_BATCH_SIZE, fewer collectives
  per step, dedup ids before the lookup to shrink the all-to-all).
* **Quality gate**: dbpedia2016 d100 runs with `EXPORT_KGREAT` queued 2026-09-04
  (`results/dbpedia_d100_e1`, `_e5`; same lr/batch/optimizer as the Wikidata
  run) — rsync the TSVs to dws and score with the GEval harness
  (`performance/kgreat_eval.md`). Then export KGrEAT TSV on the Wikidata d100 run and
  score with the existing GEval harness on dws for the §4-band comparison.
