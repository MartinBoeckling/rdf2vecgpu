"""Standalone driver: prepare a corpus, then pytorch multi_gpu fit_from_walks
(Lightning spawns DDP internally). Writes embeddings parquet to argv[1] and
prints OK. Run as a plain python process (NOT under pytest/torchrun)."""
import os, sys
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
import cudf, numpy as np
from rdf2vecgpu.config import RDF2VecConfig
from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

out = sys.argv[1]
work = os.path.dirname(out)
triples = os.path.join(work, "t.parquet")
cudf.DataFrame({
    "subject":   [f"e{i%80}" for i in range(4000)],
    "predicate": [f"p{i%5}" for i in range(4000)],
    "object":    [f"e{(i*7)%80}" for i in range(4000)],
}).to_parquet(triples)
common = dict(embedding_model="skipgram", walk_strategy="random",
              walk_depth=3, walk_number=10, vector_size=16, epochs=1)
prep = GPU_RDF2Vec(config=RDF2VecConfig(backend="pytorch", multi_gpu=False, **common))
walks_path, vocab_path = prep.prepare_corpus(prep.load_data(triples), os.path.join(work, "corpus"))
m = GPU_RDF2Vec(config=RDF2VecConfig(backend="pytorch", multi_gpu=True, **common))
m.fit_from_walks(walks_path, vocab_path)
emb = m.transform().to_pandas()
vec = [c for c in emb.columns if c not in ("token", "word")]
assert np.isfinite(emb[vec].to_numpy()).all(), "non-finite embeddings"
emb.to_parquet(out)
print("DRIVER_OK")
