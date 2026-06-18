# Embedding-quality evaluation via KGrEAT (GEval downstream tasks)

Validates that the embeddings produced by each backend are *semantically* good —
not just non-divergent — by running them through KGrEAT's downstream-task battery
(classification / regression / clustering / relatedness / analogies on the GEval
gold-standard datasets: AAUP, Cities, Forbes, Metacritic, …).

This is the real quality bar for the optimizer choice, e.g. confirming that the
torchrec **rowwiseadagrad** path matches the gensim / single-GPU baselines before
scaling to the full Wikidata truthy dump (where torchrec is mandatory — the table
exceeds one GPU).

KGrEAT lives at `/ceph/mboeckli/kgreat`; `dbpedia50k` is a small DBpedia KG with
the gold-standard mappings — ideal for a fast quality gate.

## 1. Generate embeddings (one walk corpus → 3 backends)

`bench_backends.py --export-kgreat <path>` writes the entity embeddings in KGrEAT's
format (tab-sep, no header, col 0 = entity, rest = L1-normalized vector), straight
into `kg/<id>/embedding/`. KGrEAT auto-discovers every `*.tsv` there as an
"embedding model", so all backends become directly comparable.

```bash
# dbpedia50k triples (TSV) -> parquet for load_data
python -c "import pandas as pd; df=pd.read_csv('/ceph/mboeckli/kgreat/kg/dbpedia50k/data/dbpedia50k_train.tsv',sep='\t',header=None,names=['subject','predicate','object'],dtype=str); df.to_parquet('/ceph/mboeckli/rdf2vec_bench/dbpedia50k.parquet',index=False)"

# on a 4-GPU node (conda activate rdf2vecgpu_test; export NCCL_P2P_DISABLE=1):
KGE=/ceph/mboeckli/kgreat/kg/dbpedia50k/embedding
CORPUS=/ceph/mboeckli/rdf2vec_bench/corpus/dbpedia50k
HP="--walk-depth 4 --walk-number 100 --vector-size 200 --window-size 5 --epochs 5 --negative-samples 5"

python performance/bench_backends.py --prepare --data /ceph/mboeckli/rdf2vec_bench/dbpedia50k.parquet --corpus $CORPUS --out /tmp/p.jsonl --tag dbp50k $HP
python performance/bench_backends.py --backend gensim  --corpus $CORPUS --out /tmp/k.jsonl --tag g $HP --optimizer sgd --learning-rate 0.025 --export-kgreat $KGE/rdf2vecgpu-gensim.tsv
python performance/bench_backends.py --backend pytorch --corpus $CORPUS --out /tmp/k.jsonl --tag p $HP --optimizer sgd --learning-rate 0.025 --export-kgreat $KGE/rdf2vecgpu-pytorch.tsv
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 performance/bench_backends.py --backend torchrec --corpus $CORPUS --out /tmp/k.jsonl --tag t $HP --optimizer rowwiseadagrad --learning-rate 0.05 --export-kgreat $KGE/rdf2vecgpu-torchrec-rwa.tsv
```

Note: gensim/pytorch use sgd@0.025; torchrec uses **rowwiseadagrad@0.05** (its
fused path can't clip → sgd diverges; see the `torchrec-fused-clip-noop` memory).

## 2. Run KGrEAT (uses our pre-placed embeddings; skips its own generation)

KGrEAT's manager supports **podman** (`-c podman`). Images are in the DWS GitLab
registry — `podman login` is required.

```bash
cd /ceph/mboeckli/kgreat && conda activate kgreat
podman login gitlab.dws.informatik.uni-mannheim.de:5050
python . dbpedia50k pull --stage mapping -c podman
python . dbpedia50k pull --stage task    -c podman
python . dbpedia50k run  --stage mapping -c podman   # link gold entities -> dbpedia50k local names
python . dbpedia50k run  --stage task    -c podman   # run tasks on all kg/dbpedia50k/embedding/*.tsv
```

Skipping `--stage preprocessing` avoids re-generating embeddings (we supplied
ours). To *also* get KGrEAT's own jRDF2Vec + DGL-KE (TransE/DistMult/ComplEx)
baselines, run the preprocessing stage too (needs those images + more compute).

## 3. Compare

Results land under `kg/dbpedia50k/` (see `result_analysis.ipynb`). Compare the
task scores of `rdf2vecgpu-gensim` vs `rdf2vecgpu-pytorch` vs
`rdf2vecgpu-torchrec-rwa`. If torchrec-rwa matches gensim/pytorch, the optimizer
is quality-validated for the truthy-dump scale-up.
