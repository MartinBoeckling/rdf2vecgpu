# Torchrec Sharded Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `backend="torchrec"` Word2Vec training backend with row-wise sharded embeddings (torchrec + FBGEMM + Lightning Fabric) and DCP-based checkpointing.

**Architecture:** New `embedders/torchrec_word2vec.py` module hosts a `ShardedSkipGramModel`, sharding plan, Fabric loop, and DCP save/resume helpers. `gpu_rdf2vec.py` dispatches to a new `_fit_via_torchrec` method mirroring `_fit_via_gensim`. Pre-existing walk-corpus and vocab-build code paths are reused unchanged. v1 covers skip-gram only.

**Tech Stack:** torchrec, fbgemm-gpu, `lightning.fabric.Fabric`, `torch.distributed.checkpoint` (DCP), cuDF / dask-cudf (for embedding extraction), existing pytest infrastructure (`gpu` + new `multigpu` markers).

**Spec:** `docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md`

---

## File Structure

### New files
- `src/rdf2vecgpu/embedders/torchrec_word2vec.py` — model + sampling + sharding plan + extraction + checkpoint helpers + Fabric loop (~550–800 LOC)
- `test/embedders/torchrec_word2vec_test.py` — Layer 1/2/3/4/5 tests, gated by markers

### Modified files
- `src/rdf2vecgpu/config.py` — extend `backend` Literal; add four `checkpoint_*` fields
- `src/rdf2vecgpu/gpu_rdf2vec.py` — `_fit_via_torchrec` method; dispatch in `fit()`; backend-aware return type in `transform()` / `fit_transform()`
- `pyproject.toml` — `[project.optional-dependencies].torchrec` group; register `multigpu` pytest marker
- `README.md` — install snippet + "Sharded embeddings (torchrec backend)" subsection + multi-node FS limitation note

### Reused unchanged
- `src/rdf2vecgpu/corpus/walk_corpus.py` — `MultiGPUWalkCorpus.random_walk_sequences(...)`
- `src/rdf2vecgpu/helper/functions.py` — `_generate_vocab(...)` (multi-GPU path)
- `src/rdf2vecgpu/embedders/word2vec_loader.py` — cuDF→tensor loader (thin Fabric adapter only)

---

## Important conventions used by this plan

- **TDD:** every task that introduces product code starts with a failing test (Layer 1/2 tests run on the GPU runner; Layer 3/4/5 tests are `multigpu`-marked and must be run manually with `torchrun --standalone --nproc_per_node=2 -m pytest -m multigpu`).
- **Commits:** one commit per task at minimum; intermediate test-passes within a task may also commit.
- **Markers:** `pytest.mark.gpu` for single-GPU tests; new `pytest.mark.multigpu` for tests that require `WORLD_SIZE >= 2`.
- **Lazy imports:** any code path that imports `torchrec` or `fbgemm_gpu` is reached only after the pre-flight ImportError check passes — tests in Layer 1 must run without those packages installed.

---

## Task 1: Project plumbing — pyproject extras + multigpu marker

**Files:**
- Modify: `pyproject.toml:36-57`

- [ ] **Step 1: Add the torchrec extras group**

Edit `pyproject.toml`, in the `[project.optional-dependencies]` block, add after the `gensim` group:

```toml
torchrec = [
    "torchrec>=0.6.0",
    "fbgemm-gpu>=0.6.0",
]
```

- [ ] **Step 2: Register the multigpu pytest marker**

Edit `pyproject.toml`, in the `[tool.pytest.ini_options]` `markers` list, append:

```toml
    "multigpu: tests that require WORLD_SIZE>=2; run with `torchrun --standalone --nproc_per_node=2 -m pytest -m multigpu`",
```

- [ ] **Step 3: Verify pyproject parses**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`
Expected: no output (no exception)

- [ ] **Step 4: Verify pytest sees the marker**

Run: `pytest --markers | grep multigpu`
Expected: a line beginning with `@pytest.mark.multigpu`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build(pyproject): add torchrec extras + multigpu pytest marker"
```

---

## Task 2: Config extensions (TDD)

**Files:**
- Modify: `src/rdf2vecgpu/config.py`
- Test: `test/config_test.py`

- [ ] **Step 1: Read existing config tests for style**

Run: `head -80 test/config_test.py`
Note: how `RDF2VecConfig` is instantiated, how defaults are asserted.

- [ ] **Step 2: Write the failing tests**

Append to `test/config_test.py`:

```python
def test_backend_accepts_torchrec_value():
    cfg = RDF2VecConfig(backend="torchrec")
    assert cfg.backend == "torchrec"


def test_checkpoint_defaults_match_spec():
    cfg = RDF2VecConfig()
    assert cfg.checkpoint_dir is None
    assert cfg.checkpoint_interval_minutes == 30
    assert cfg.checkpoint_overwrite is False
    assert cfg.checkpoint_keep_last == 3


def test_checkpoint_dir_accepts_string():
    cfg = RDF2VecConfig(checkpoint_dir="/tmp/ckpt")
    assert cfg.checkpoint_dir == "/tmp/ckpt"
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `pytest test/config_test.py -k "torchrec or checkpoint" -v`
Expected: FAIL — `backend` Literal rejects `"torchrec"`; `checkpoint_*` attributes don't exist.

- [ ] **Step 4: Extend the Literal and add fields**

Edit `src/rdf2vecgpu/config.py`. Find the `backend` field declaration (`backend: Literal["pytorch", "gensim"] = "pytorch"`) and change it to:

```python
    backend: Literal["pytorch", "gensim", "torchrec"] = "pytorch"
```

Then, after the `backend` field, add (with surrounding spacing matching the file's style):

```python
    # Checkpointing — only consumed by backend="torchrec". Ignored by the
    # "pytorch" and "gensim" backends.
    checkpoint_dir: str | None = Field(default=None)
    checkpoint_interval_minutes: int = Field(default=30, ge=1)
    checkpoint_overwrite: bool = Field(default=False)
    checkpoint_keep_last: int = Field(default=3, ge=1)
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `pytest test/config_test.py -k "torchrec or checkpoint" -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rdf2vecgpu/config.py test/config_test.py
git commit -m "config: add backend='torchrec' + checkpoint_* fields"
```

---

## Task 3: Pre-flight validation + dispatch stub (Layer 1 tests)

**Files:**
- Modify: `src/rdf2vecgpu/gpu_rdf2vec.py` (around the `fit()` dispatch, ~line 700)
- Create: `test/embedders/torchrec_word2vec_test.py`

- [ ] **Step 1: Create the test file with import-time skip + Layer 1 tests**

Create `test/embedders/torchrec_word2vec_test.py`:

```python
"""Tests for the torchrec sharded-embedding backend.

Layer 1 tests (this section) run without torchrec/fbgemm installed and
without a GPU — they exercise pre-flight config validation only.
Layers 2+ require either the `gpu` or `multigpu` marker.
"""
from __future__ import annotations

import pytest

from rdf2vecgpu.config import RDF2VecConfig
from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec


# ───────────────────────────── Layer 1 ────────────────────────────────


def test_torchrec_backend_with_cbow_raises_not_implemented():
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="cbow", multi_gpu=True)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(NotImplementedError, match="skipgram"):
        # No edge_df needed — the pre-flight check fires before walk gen.
        model.fit(edge_df=None)


def test_torchrec_backend_without_multi_gpu_raises_value_error():
    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=False)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(ValueError, match="multi_gpu=True"):
        model.fit(edge_df=None)


def test_torchrec_backend_without_distributed_init_raises_runtime_error():
    """No torchrun → torch.distributed.is_initialized() is False."""
    import torch.distributed as dist

    assert not dist.is_initialized(), "this test must run outside a distributed launch"

    cfg = RDF2VecConfig(backend="torchrec", embedding_model="skipgram", multi_gpu=True)
    model = GPU_RDF2Vec(config=cfg)
    with pytest.raises(RuntimeError, match="torchrun"):
        model.fit(edge_df=None)
```

- [ ] **Step 2: Run the new tests — verify they fail**

Run: `pytest test/embedders/torchrec_word2vec_test.py -v`
Expected: all three FAIL — the dispatch doesn't know about `backend="torchrec"` yet, so it falls through to a different error.

- [ ] **Step 3: Add the pre-flight + dispatch stub in `gpu_rdf2vec.py`**

Edit `src/rdf2vecgpu/gpu_rdf2vec.py`. Find the existing `if self.config.backend == "gensim":` block (around line 703). **Before** that `if`, add:

```python
        if self.config.backend == "torchrec":
            self._fit_via_torchrec(edge_df, walk_vertices)
            return
```

Then, find `_fit_via_gensim` (around line 761) and **after** its closing line, add a new method:

```python
    def _fit_via_torchrec(
        self, edge_df, walk_vertices: cudf.Series | None = None
    ) -> None:
        """Train the sharded skip-gram model via torchrec + Fabric + DCP.

        Pre-flight checks run before any walk generation so the user
        sees configuration errors immediately rather than mid-run.
        See ``docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md``
        §5 for the full error-handling contract.
        """
        # Pre-flight: model coverage
        if self.config.embedding_model != "skipgram":
            raise NotImplementedError(
                f"backend='torchrec' currently supports embedding_model='skipgram' "
                f"only (got {self.config.embedding_model!r})."
            )

        # Pre-flight: multi-GPU is mandatory
        if not self.config.multi_gpu:
            raise ValueError(
                "backend='torchrec' requires multi_gpu=True "
                "(the sharded embedding path is multi-process by construction)."
            )

        # Pre-flight: imports
        try:
            import torchrec  # noqa: F401
            import fbgemm_gpu  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='torchrec' requires the torchrec extra. "
                "Install with: pip install rdf2vecgpu[torchrec]"
            ) from e

        # Pre-flight: distributed init
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError(
                "backend='torchrec' requires a distributed launch. "
                "Run via: torchrun --standalone --nproc_per_node=N your_script.py"
            )

        # Stubbed for later tasks — wiring lives in
        # rdf2vecgpu.embedders.torchrec_word2vec.train_torchrec_word2vec
        from rdf2vecgpu.embedders.torchrec_word2vec import train_torchrec_word2vec
        train_torchrec_word2vec(self, edge_df, walk_vertices)
```

- [ ] **Step 4: Create an empty `torchrec_word2vec.py` so the import in the dispatch resolves**

Create `src/rdf2vecgpu/embedders/torchrec_word2vec.py`:

```python
"""Torchrec-backed sharded Word2Vec training.

See docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md
for the design. v1 covers skip-gram only.

All torchrec / fbgemm imports happen *inside* function bodies so this
module can be imported without those extras installed (the import-time
check in :func:`train_torchrec_word2vec` is what surfaces the error).
"""
from __future__ import annotations


def train_torchrec_word2vec(model, edge_df, walk_vertices):
    raise NotImplementedError(
        "train_torchrec_word2vec is a stub — implemented in Task 11."
    )
```

- [ ] **Step 5: Run Layer 1 tests, verify they pass**

Run: `pytest test/embedders/torchrec_word2vec_test.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rdf2vecgpu/gpu_rdf2vec.py src/rdf2vecgpu/embedders/torchrec_word2vec.py test/embedders/torchrec_word2vec_test.py
git commit -m "feat(torchrec): pre-flight checks + dispatch stub for backend='torchrec'"
```

---

## Task 4: `ShardedSkipGramModel` skeleton (Layer 2, single-process degenerate-shard)

This task builds the module shell with two `EmbeddingCollection`s in/out and verifies it can be instantiated under a degenerate `world_size=1` (table-wise) plan. Forward + loss come in Task 6.

**Files:**
- Modify: `src/rdf2vecgpu/embedders/torchrec_word2vec.py`
- Modify: `test/embedders/torchrec_word2vec_test.py` (Layer 2 section)

- [ ] **Step 1: Write the failing test**

Append to `test/embedders/torchrec_word2vec_test.py`:

```python
# ───────────────────────────── Layer 2 ────────────────────────────────


torchrec = pytest.importorskip("torchrec")
torch_lib = pytest.importorskip("torch")


@pytest.mark.gpu
def test_sharded_skipgram_model_constructs_in_degenerate_world_size_one(tmp_path):
    """world_size=1 → planner degenerates to table-wise on rank 0.

    This test verifies the module wiring without needing a real
    multi-process launch.
    """
    import os
    import torch
    import torch.distributed as dist

    # Single-process distributed init on localhost (gloo backend is fine —
    # we don't need NCCL for module construction).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import ShardedSkipGramModel

        model = ShardedSkipGramModel(vocab_size=1024, embedding_dim=16)
        sd = model.state_dict()
        # Both in and out tables present.
        assert any("in_embeddings" in k for k in sd), sd.keys()
        assert any("out_embeddings" in k for k in sd), sd.keys()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest test/embedders/torchrec_word2vec_test.py::test_sharded_skipgram_model_constructs_in_degenerate_world_size_one -v --gpu`
Expected: FAIL — `ShardedSkipGramModel` doesn't exist.

- [ ] **Step 3: Implement the model skeleton**

Edit `src/rdf2vecgpu/embedders/torchrec_word2vec.py`. Replace the entire module body with:

```python
"""Torchrec-backed sharded Word2Vec training.

See docs/superpowers/specs/2026-05-26-torchrec-sharded-embeddings-design.md
for the design. v1 covers skip-gram only.

All torchrec / fbgemm imports happen *inside* function/class bodies so
this module can be imported without those extras installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    pass


class ShardedSkipGramModel(nn.Module):
    """Skip-gram with two row-wise sharded embedding tables (in + out).

    Sharding is applied later by wrapping this module in
    ``torchrec.distributed.DistributedModelParallel`` with a plan from
    :func:`_build_sharding_plan`. Prior to wrapping, the table parameters
    live on whichever device the caller placed them.
    """

    IN_TABLE_NAME = "in_embeddings"
    OUT_TABLE_NAME = "out_embeddings"

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        from torchrec import EmbeddingCollection, EmbeddingConfig

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        in_cfg = EmbeddingConfig(
            name=self.IN_TABLE_NAME,
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            feature_names=["center"],
        )
        out_cfg = EmbeddingConfig(
            name=self.OUT_TABLE_NAME,
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            feature_names=["context_or_negative"],
        )
        self.in_collection = EmbeddingCollection(tables=[in_cfg])
        self.out_collection = EmbeddingCollection(tables=[out_cfg])


def train_torchrec_word2vec(model, edge_df, walk_vertices):
    raise NotImplementedError(
        "train_torchrec_word2vec is a stub — implemented in Task 11."
    )
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest test/embedders/torchrec_word2vec_test.py::test_sharded_skipgram_model_constructs_in_degenerate_world_size_one -v --gpu`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/embedders/torchrec_word2vec.py test/embedders/torchrec_word2vec_test.py
git commit -m "feat(torchrec): ShardedSkipGramModel skeleton (in/out EmbeddingCollections)"
```

---

## Task 5: Negative sampling (`_sample_negatives`)

**Files:**
- Modify: `src/rdf2vecgpu/embedders/torchrec_word2vec.py`
- Modify: `test/embedders/torchrec_word2vec_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/embedders/torchrec_word2vec_test.py`:

```python
def test_sample_negatives_uniform_shape_and_range():
    from rdf2vecgpu.embedders.torchrec_word2vec import _sample_negatives

    out = _sample_negatives(
        n_rows=64,
        n_neg=5,
        vocab_size=1024,
        device=torch.device("cpu"),
        strategy="uniform",
        token_counts=None,
    )
    assert out.shape == (64, 5)
    assert out.dtype == torch.long
    assert int(out.min()) >= 0
    assert int(out.max()) < 1024


def test_sample_negatives_unigram_uses_token_counts():
    from rdf2vecgpu.embedders.torchrec_word2vec import _sample_negatives

    # All probability mass on a single token (idx=7).
    counts = torch.zeros(1024)
    counts[7] = 1.0
    out = _sample_negatives(
        n_rows=128,
        n_neg=3,
        vocab_size=1024,
        device=torch.device("cpu"),
        strategy="unigram",
        token_counts=counts,
    )
    # With all mass on idx=7, every drawn negative must be 7.
    assert torch.all(out == 7)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest test/embedders/torchrec_word2vec_test.py -k sample_negatives -v`
Expected: FAIL — `_sample_negatives` doesn't exist.

- [ ] **Step 3: Implement `_sample_negatives`**

Edit `src/rdf2vecgpu/embedders/torchrec_word2vec.py`. After the imports and before `class ShardedSkipGramModel`, add:

```python
def _sample_negatives(
    n_rows: int,
    n_neg: int,
    vocab_size: int,
    device: torch.device,
    strategy: str = "unigram",
    token_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample ``(n_rows, n_neg)`` negative token ids.

    Mirrors the f^0.75 unigram sampler used by the existing pytorch
    backend (see ``embedders/word2vec.py::_sample_negatives``). Returns
    a flat ``long`` tensor; the caller is responsible for routing
    through the sharded output embedding.
    """
    if strategy == "uniform":
        return torch.randint(
            low=0, high=vocab_size, size=(n_rows, n_neg),
            dtype=torch.long, device=device,
        )

    if strategy == "unigram":
        if token_counts is None:
            raise ValueError("strategy='unigram' requires token_counts")
        weights = token_counts.to(device=device, dtype=torch.float32).pow(0.75)
        # torch.multinomial replaces=True is fine for negative sampling
        # (rare duplicates within a row are negligible at typical
        # neg_samples ≤ 25 against vocab_size in the millions).
        return torch.multinomial(
            weights, num_samples=n_rows * n_neg, replacement=True,
        ).view(n_rows, n_neg)

    raise ValueError(f"unknown negative-sampling strategy: {strategy!r}")
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest test/embedders/torchrec_word2vec_test.py -k sample_negatives -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/embedders/torchrec_word2vec.py test/embedders/torchrec_word2vec_test.py
git commit -m "feat(torchrec): _sample_negatives — uniform + f^0.75 unigram"
```

---

## Task 6: `ShardedSkipGramModel.forward` + loss

**Files:**
- Modify: `src/rdf2vecgpu/embedders/torchrec_word2vec.py`
- Modify: `test/embedders/torchrec_word2vec_test.py`

- [ ] **Step 1: Write the failing test**

Append to `test/embedders/torchrec_word2vec_test.py`:

```python
@pytest.mark.gpu
def test_sharded_skipgram_forward_returns_finite_loss(tmp_path):
    import os
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _sample_negatives,
        )

        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=1024, embedding_dim=16).to(device)

        batch = 32
        center = torch.randint(0, 1024, (batch,), device=device)
        context = torch.randint(0, 1024, (batch,), device=device)
        negative = _sample_negatives(
            n_rows=batch, n_neg=5, vocab_size=1024,
            device=device, strategy="uniform", token_counts=None,
        )

        loss = model(center, context, negative)
        assert loss.dim() == 0
        assert torch.isfinite(loss).item()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest test/embedders/torchrec_word2vec_test.py::test_sharded_skipgram_forward_returns_finite_loss -v --gpu`
Expected: FAIL — `forward` is not defined on `ShardedSkipGramModel`.

- [ ] **Step 3: Implement `forward`**

Edit `src/rdf2vecgpu/embedders/torchrec_word2vec.py`. Inside `class ShardedSkipGramModel`, after `__init__`, add:

```python
    def forward(
        self,
        center: torch.Tensor,   # (B,) long
        context: torch.Tensor,  # (B,) long
        negative: torch.Tensor, # (B, neg) long
    ) -> torch.Tensor:
        """Skip-gram negative-sampling loss.

        Mirrors :class:`rdf2vecgpu.embedders.word2vec.Word2VecSkipgram.forward`:
        positive logit = <in[center], out[context]>;
        negative logits = <in[center], out[neg]>;
        loss = -log σ(pos) - Σ log σ(-neg_k).
        """
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        batch_size = center.size(0)
        n_neg = negative.size(1)

        # Build KJTs for the two collections. Each row contributes one id
        # for the in-table (center) and (1 + n_neg) ids for the out-table
        # (one positive context + n_neg negatives).
        in_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["center"],
            values=center,
            lengths=torch.ones(batch_size, dtype=torch.long, device=center.device),
        )
        out_values = torch.cat([context.unsqueeze(1), negative], dim=1).reshape(-1)
        out_lengths = torch.full(
            (batch_size,), 1 + n_neg, dtype=torch.long, device=center.device,
        )
        out_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["context_or_negative"],
            values=out_values,
            lengths=out_lengths,
        )

        in_dict = self.in_collection(in_kjt)
        out_dict = self.out_collection(out_kjt)

        center_embed = in_dict["center"].values().view(batch_size, self.embedding_dim)
        out_embed = (
            out_dict["context_or_negative"].values().view(batch_size, 1 + n_neg, self.embedding_dim)
        )

        pos_embed = out_embed[:, 0, :]                # (B, dim)
        neg_embed = out_embed[:, 1:, :]               # (B, neg, dim)

        pos_score = (center_embed * pos_embed).sum(dim=-1)                            # (B,)
        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(-1)).squeeze(-1)      # (B, neg)

        pos_loss = -torch.nn.functional.logsigmoid(pos_score)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=-1)
        return (pos_loss + neg_loss).mean()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest test/embedders/torchrec_word2vec_test.py::test_sharded_skipgram_forward_returns_finite_loss -v --gpu`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/embedders/torchrec_word2vec.py test/embedders/torchrec_word2vec_test.py
git commit -m "feat(torchrec): ShardedSkipGramModel.forward — negative-sampling skip-gram loss"
```

---

## Task 7: `_build_sharding_plan` + DMP wrapping helper

**Files:**
- Modify: `src/rdf2vecgpu/embedders/torchrec_word2vec.py`
- Modify: `test/embedders/torchrec_word2vec_test.py`

- [ ] **Step 1: Write the failing test**

Append to `test/embedders/torchrec_word2vec_test.py`:

```python
@pytest.mark.gpu
def test_build_sharding_plan_returns_plan_with_both_tables():
    import os
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _build_sharding_plan,
        )
        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=1024, embedding_dim=16).to(device)
        plan = _build_sharding_plan(model, world_size=1, device=device)
        # The plan should reference both tables by name.
        plan_str = str(plan)
        assert ShardedSkipGramModel.IN_TABLE_NAME in plan_str
        assert ShardedSkipGramModel.OUT_TABLE_NAME in plan_str
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest test/embedders/torchrec_word2vec_test.py::test_build_sharding_plan_returns_plan_with_both_tables -v --gpu`
Expected: FAIL — `_build_sharding_plan` doesn't exist.

- [ ] **Step 3: Implement `_build_sharding_plan`**

Edit `src/rdf2vecgpu/embedders/torchrec_word2vec.py`. After `_sample_negatives` (or wherever module-level helpers live), add:

```python
def _build_sharding_plan(model: nn.Module, world_size: int, device: torch.device):
    """Defer to torchrec's planner; no constraints in v1.

    The planner will pick row-wise sharding automatically for tables
    that are large enough to benefit from it. Smaller tables may end up
    table-wise on a single rank, which is fine — the user's reason for
    choosing this backend is the *large* tables.
    """
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.types import ModuleSharder
    from torchrec.distributed.embedding import EmbeddingCollectionSharder

    topology = Topology(
        world_size=world_size,
        compute_device=device.type,  # "cuda"
    )
    planner = EmbeddingShardingPlanner(topology=topology)
    sharders: list[ModuleSharder] = [EmbeddingCollectionSharder()]
    return planner.plan(model, sharders=sharders)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest test/embedders/torchrec_word2vec_test.py::test_build_sharding_plan_returns_plan_with_both_tables -v --gpu`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/embedders/torchrec_word2vec.py test/embedders/torchrec_word2vec_test.py
git commit -m "feat(torchrec): _build_sharding_plan via EmbeddingShardingPlanner"
```

---

## Task 8: `extract_embeddings_to_dask_cudf`

**Files:**
- Modify: `src/rdf2vecgpu/embedders/torchrec_word2vec.py`
- Modify: `test/embedders/torchrec_word2vec_test.py`

- [ ] **Step 1: Write the failing test**

Append to `test/embedders/torchrec_word2vec_test.py`:

```python
@pytest.mark.gpu
def test_extract_embeddings_to_dask_cudf_schema_and_row_count():
    import os
    import torch
    import torch.distributed as dist
    import cudf

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            extract_embeddings_to_dask_cudf,
        )
        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=128, embedding_dim=8).to(device)

        # Build a tiny vocab_df mapping token_id -> entity (str).
        vocab_df = cudf.DataFrame({
            "token_id": cudf.Series(range(128), dtype="int64"),
            "entity":   cudf.Series([f"e{i}" for i in range(128)]),
        })

        out = extract_embeddings_to_dask_cudf(model, vocab_df)
        out_pdf = out.compute().to_pandas()

        assert len(out_pdf) == 128
        # Required columns: token_id, entity, and one column per embedding dim
        assert "token_id" in out_pdf.columns
        assert "entity" in out_pdf.columns
        for d in range(8):
            assert str(d) in out_pdf.columns, f"missing dim column {d}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest test/embedders/torchrec_word2vec_test.py::test_extract_embeddings_to_dask_cudf_schema_and_row_count -v --gpu`
Expected: FAIL — `extract_embeddings_to_dask_cudf` doesn't exist.

- [ ] **Step 3: Implement `extract_embeddings_to_dask_cudf`**

Edit `src/rdf2vecgpu/embedders/torchrec_word2vec.py`. Add:

```python
def extract_embeddings_to_dask_cudf(model, vocab_df):
    """Return embeddings as a ``dask_cudf.DataFrame`` (one partition per rank).

    Reads the *local* shard of the in-embedding table on each rank;
    joins it shard-locally with the vocab slice for that token-id
    range; wraps each rank's cudf.DataFrame as a single dask partition.
    No rank-0 gather.

    The output schema is::

        token_id  int64
        entity    string
        "0"       float32
        "1"       float32
        ...
        str(dim-1) float32
    """
    import cudf
    import cupy
    import dask_cudf
    import torch.distributed as dist
    from dask import delayed

    # Locate the in-embedding table parameter on this rank.
    # Under DMP, the parameter is sharded; .local_tensor() returns the
    # rank-local slab. Before DMP wrapping (degenerate world_size=1), we
    # fall back to .data on the underlying nn.Parameter.
    in_param = _get_in_table_local_tensor(model)         # cuda tensor (local_rows, dim)
    local_rows, dim = in_param.shape

    # Compute the (start, end) row range this rank owns.
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    start, end = _local_row_range(model.vocab_size, world_size, rank)
    assert end - start == local_rows, (
        f"local rows {local_rows} ≠ shard size {end - start} on rank {rank}"
    )

    # Move to a cudf.DataFrame with dim columns labeled "0".."dim-1".
    cupy_arr = cupy.asarray(in_param.detach())
    shard_df = cudf.DataFrame({str(d): cupy_arr[:, d] for d in range(dim)})
    shard_df["token_id"] = cudf.Series(range(start, end), dtype="int64")

    # Shard-local join with the vocab slice for [start, end).
    vocab_slice = vocab_df[(vocab_df["token_id"] >= start) & (vocab_df["token_id"] < end)]
    shard_df = shard_df.merge(vocab_slice, on="token_id", how="left")

    # Wrap each rank's cudf as one dask partition. With world_size=1
    # this is a single-partition dask_cudf; in true multi-rank runs
    # each rank produces its own partition and dask glues them.
    return dask_cudf.from_cudf(shard_df, npartitions=1)


def _local_row_range(vocab_size: int, world_size: int, rank: int) -> tuple[int, int]:
    """Even split with the remainder absorbed by the last rank."""
    base = vocab_size // world_size
    start = rank * base
    end = vocab_size if rank == world_size - 1 else start + base
    return start, end


def _get_in_table_local_tensor(model) -> torch.Tensor:
    """Return the rank-local tensor for the in-embedding table.

    Handles both the pre-DMP nn.Module case (`in_collection.embeddings`
    holds a vanilla nn.Embedding) and the post-DMP sharded case
    (`local_tensor()` exposes the rank-local slab).
    """
    # Post-DMP: torchrec exposes sharded params via a state_dict where
    # values are ShardedTensor or DTensor with `.local_tensor()`.
    try:
        sd = model.state_dict()
        for key, val in sd.items():
            if ShardedSkipGramModel.IN_TABLE_NAME in key and "weight" in key:
                if hasattr(val, "local_tensor"):
                    return val.local_tensor()
                return val
    except Exception:
        pass

    # Fallback: pre-DMP, walk the EmbeddingCollection's underlying table.
    for table in model.in_collection.embeddings.values():
        return table.weight.data
    raise RuntimeError("could not locate in-embedding table on this rank")
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest test/embedders/torchrec_word2vec_test.py::test_extract_embeddings_to_dask_cudf_schema_and_row_count -v --gpu`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/embedders/torchrec_word2vec.py test/embedders/torchrec_word2vec_test.py
git commit -m "feat(torchrec): extract_embeddings_to_dask_cudf — per-rank shard extraction"
```

---

## Task 9: `_save_checkpoint` (DCP write + metadata + latest symlink + pruning)

**Files:**
- Modify: `src/rdf2vecgpu/embedders/torchrec_word2vec.py`
- Modify: `test/embedders/torchrec_word2vec_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/embedders/torchrec_word2vec_test.py`:

```python
@pytest.mark.gpu
def test_save_checkpoint_writes_step_dir_metadata_and_latest_symlink(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
        )
        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(
            model, opt,
            epoch=1, global_step=500,
            checkpoint_dir=str(ckpt_dir),
            keep_last=3,
            vocab_path="/data/vocab.parquet",
            config_snapshot={"backend": "torchrec"},
        )
        # step directory created
        assert (ckpt_dir / "step_00000500").is_dir()
        # metadata.pt at root, with the right keys
        assert (ckpt_dir / "metadata.pt").is_file()
        meta = torch.load(ckpt_dir / "metadata.pt", weights_only=False)
        assert meta["epoch"] == 1
        assert meta["global_step"] == 500
        assert meta["vocab_path"] == "/data/vocab.parquet"
        # latest symlink resolves to the step directory
        latest = ckpt_dir / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == (ckpt_dir / "step_00000500").resolve()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_save_checkpoint_prunes_old_step_dirs(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
        )
        device = torch.device("cuda")
        model = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"

        for step in (100, 200, 300, 400, 500):
            _save_checkpoint(
                model, opt, epoch=0, global_step=step,
                checkpoint_dir=str(ckpt_dir), keep_last=3,
                vocab_path="/v", config_snapshot={},
            )
        kept = sorted(p.name for p in ckpt_dir.iterdir() if p.name.startswith("step_"))
        assert kept == ["step_00000300", "step_00000400", "step_00000500"]
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest test/embedders/torchrec_word2vec_test.py -k save_checkpoint -v --gpu`
Expected: FAIL — `_save_checkpoint` doesn't exist.

- [ ] **Step 3: Implement `_save_checkpoint`**

Edit `src/rdf2vecgpu/embedders/torchrec_word2vec.py`. Add at module level:

```python
def _save_checkpoint(
    model,
    optimizer,
    epoch: int,
    global_step: int,
    checkpoint_dir: str,
    keep_last: int,
    vocab_path: str,
    config_snapshot: dict,
) -> None:
    """Write a DCP shard set + rank-0 metadata.pt + update ``latest`` symlink.

    Pruning policy: keep the most recent ``keep_last`` ``step_*``
    directories; everything else is removed.

    Layout (see spec §4.4):
        {checkpoint_dir}/
          step_<global_step:08d>/    # DCP shards, written by all ranks
          metadata.pt                # rank-0 only
          latest -> step_*
    """
    import os
    import shutil
    from pathlib import Path

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    ckpt_root = Path(checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    step_dir = ckpt_root / f"step_{global_step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # DCP save: all ranks participate.
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    dcp.save(
        state_dict=state_dict,
        storage_writer=dcp.FileSystemWriter(str(step_dir)),
    )

    # Rank-0-only: metadata.pt + latest symlink + pruning.
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        meta = {
            "epoch": epoch,
            "global_step": global_step,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "vocab_path": vocab_path,
            "config_snapshot": config_snapshot,
        }
        torch.save(meta, ckpt_root / "metadata.pt")

        # Atomic symlink update via rename.
        tmp_link = ckpt_root / "latest.tmp"
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        os.symlink(step_dir.name, tmp_link)         # relative target
        os.replace(tmp_link, ckpt_root / "latest")

        # Prune: keep the most recent `keep_last` step_* directories.
        step_dirs = sorted(
            [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")],
            key=lambda p: p.name,
        )
        for stale in step_dirs[:-keep_last]:
            shutil.rmtree(stale)

    # Barrier so other ranks don't race ahead of rank-0 housekeeping.
    if dist.is_initialized():
        dist.barrier()
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest test/embedders/torchrec_word2vec_test.py -k save_checkpoint -v --gpu`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/embedders/torchrec_word2vec.py test/embedders/torchrec_word2vec_test.py
git commit -m "feat(torchrec): _save_checkpoint — DCP shard set + metadata + latest symlink + pruning"
```

---

## Task 10: `_maybe_resume_checkpoint` (round-trip + guards)

**Files:**
- Modify: `src/rdf2vecgpu/embedders/torchrec_word2vec.py`
- Modify: `test/embedders/torchrec_word2vec_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/embedders/torchrec_word2vec_test.py`:

```python
@pytest.mark.gpu
def test_resume_round_trip_restores_weights_and_step(tmp_path):
    import os, copy, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
            _maybe_resume_checkpoint,
        )
        device = torch.device("cuda")

        # Save
        m1 = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(
            m1, opt1, epoch=3, global_step=12345,
            checkpoint_dir=str(ckpt_dir), keep_last=3,
            vocab_path="/v", config_snapshot={},
        )
        snapshot = copy.deepcopy(m1.state_dict())

        # Resume into a fresh model.
        m2 = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
        start_epoch, start_step = _maybe_resume_checkpoint(
            m2, opt2,
            checkpoint_dir=str(ckpt_dir), overwrite=False, current_vocab_path="/v",
        )
        assert start_epoch == 3
        assert start_step == 12345
        for k, v in snapshot.items():
            torch.testing.assert_close(m2.state_dict()[k], v)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_resume_missing_dir_returns_fresh(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _maybe_resume_checkpoint,
        )
        device = torch.device("cuda")
        m = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        start_epoch, start_step = _maybe_resume_checkpoint(
            m, opt,
            checkpoint_dir=str(tmp_path / "does_not_exist"),
            overwrite=False, current_vocab_path="/v",
        )
        assert (start_epoch, start_step) == (0, 0)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_resume_with_vocab_mismatch_raises(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
            _maybe_resume_checkpoint,
        )
        device = torch.device("cuda")
        m = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(
            m, opt, epoch=0, global_step=1,
            checkpoint_dir=str(ckpt_dir), keep_last=3,
            vocab_path="/vocab_A.parquet", config_snapshot={},
        )
        with pytest.raises(RuntimeError, match="vocab"):
            _maybe_resume_checkpoint(
                m, opt,
                checkpoint_dir=str(ckpt_dir),
                overwrite=False,
                current_vocab_path="/vocab_B.parquet",
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
def test_resume_overwrite_wipes_dir_and_returns_fresh(tmp_path):
    import os, torch, torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        from rdf2vecgpu.embedders.torchrec_word2vec import (
            ShardedSkipGramModel,
            _save_checkpoint,
            _maybe_resume_checkpoint,
        )
        device = torch.device("cuda")
        m = ShardedSkipGramModel(vocab_size=64, embedding_dim=4).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        ckpt_dir = tmp_path / "ckpt"
        _save_checkpoint(
            m, opt, epoch=2, global_step=999,
            checkpoint_dir=str(ckpt_dir), keep_last=3,
            vocab_path="/v", config_snapshot={},
        )
        start_epoch, start_step = _maybe_resume_checkpoint(
            m, opt,
            checkpoint_dir=str(ckpt_dir), overwrite=True, current_vocab_path="/v",
        )
        assert (start_epoch, start_step) == (0, 0)
        # No step_* dirs should remain
        assert not any(p.name.startswith("step_") for p in ckpt_dir.iterdir())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest test/embedders/torchrec_word2vec_test.py -k "resume" -v --gpu`
Expected: 4 FAILs — `_maybe_resume_checkpoint` doesn't exist.

- [ ] **Step 3: Implement `_maybe_resume_checkpoint`**

Edit `src/rdf2vecgpu/embedders/torchrec_word2vec.py`. Add:

```python
def _maybe_resume_checkpoint(
    model,
    optimizer,
    checkpoint_dir: str | None,
    overwrite: bool,
    current_vocab_path: str,
) -> tuple[int, int]:
    """Resume from ``{checkpoint_dir}/latest`` if it exists, else fresh.

    Returns ``(start_epoch, start_step)``. See spec §4.4 for the full
    decision table.
    """
    import shutil
    from pathlib import Path

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    if checkpoint_dir is None:
        return 0, 0

    ckpt_root = Path(checkpoint_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0

    if overwrite and ckpt_root.exists():
        if rank == 0:
            shutil.rmtree(ckpt_root)
            ckpt_root.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return 0, 0

    if not ckpt_root.exists():
        if rank == 0:
            ckpt_root.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return 0, 0

    latest = ckpt_root / "latest"
    if not latest.exists():
        # Empty dir, or a never-completed save — start fresh in place.
        return 0, 0

    step_dir = latest.resolve()
    if not step_dir.is_dir():
        raise RuntimeError(
            f"Checkpoint at {step_dir} is missing or corrupt. Pass "
            "checkpoint_overwrite=True to start fresh, or restore the "
            "checkpoint contents."
        )

    # Vocab-mismatch guard.
    meta = torch.load(ckpt_root / "metadata.pt", weights_only=False)
    saved_vocab_path = meta["vocab_path"]
    if saved_vocab_path != current_vocab_path:
        raise RuntimeError(
            f"Cannot resume: saved checkpoint was built against vocab at "
            f"{saved_vocab_path!r}, but current run resolved vocab to "
            f"{current_vocab_path!r}. Vocab IDs would no longer match. Pass "
            "checkpoint_overwrite=True to start fresh."
        )

    # DCP load into the already-instantiated model/optimizer.
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    dcp.load(
        state_dict=state_dict,
        storage_reader=dcp.FileSystemReader(str(step_dir)),
    )
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])

    # Restore RNG state on the loading rank only — DCP doesn't reshard
    # RNG and we accept this small reproducibility caveat (documented).
    if rank == 0:
        torch.set_rng_state(meta["rng_state"])
        torch.cuda.set_rng_state_all(meta["cuda_rng_state_all"])

    if dist.is_initialized():
        dist.barrier()

    return int(meta["epoch"]), int(meta["global_step"])
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest test/embedders/torchrec_word2vec_test.py -k "resume" -v --gpu`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/embedders/torchrec_word2vec.py test/embedders/torchrec_word2vec_test.py
git commit -m "feat(torchrec): _maybe_resume_checkpoint — DCP load + vocab guard + overwrite"
```

---

## Task 11: `train_torchrec_word2vec` — Fabric loop tying it all together

This is the integration task. It produces no new test of its own; correctness is exercised by Layer 3/4/5 tests in Task 13.

**Files:**
- Modify: `src/rdf2vecgpu/embedders/torchrec_word2vec.py`

- [ ] **Step 1: Replace the `train_torchrec_word2vec` stub with the full loop**

Edit `src/rdf2vecgpu/embedders/torchrec_word2vec.py`. Replace the stub `train_torchrec_word2vec` with:

```python
def train_torchrec_word2vec(rdf2vec, edge_df, walk_vertices) -> None:
    """Top-level training entry point for the torchrec backend.

    Called from :meth:`GPU_RDF2Vec._fit_via_torchrec` once pre-flight
    checks have passed. ``rdf2vec`` is the :class:`GPU_RDF2Vec`
    instance, which owns the config, the tracker, and the walk-corpus
    machinery.
    """
    import time
    from lightning.fabric import Fabric
    import torch
    import torch.distributed as dist
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.optim.optimizers import (
        in_backward_optimizer_filter,
    )
    from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward

    cfg = rdf2vec.config

    # 1. Walk generation — backend-agnostic; uses existing multi-GPU path.
    walks_path = rdf2vec.walk_generation_sequences(
        edge_df=edge_df, walk_vertices=walk_vertices,
    )

    # 2. Vocab build — reuse the existing multi-GPU helper.
    vocab_df, token_counts, vocab_path = rdf2vec._build_vocab_for_walks(walks_path)
    vocab_size = len(vocab_df)

    # 3. Fabric.
    fabric = Fabric(
        accelerator="gpu",
        devices="auto",
        precision=_fabric_precision_from_config(cfg),
    )
    fabric.launch()

    # 4. Build the sharded model + plan + DMP wrap.
    model = ShardedSkipGramModel(
        vocab_size=vocab_size, embedding_dim=cfg.vector_size,
    ).to(fabric.device)
    plan = _build_sharding_plan(model, world_size=fabric.world_size, device=fabric.device)
    model = DistributedModelParallel(
        module=model, device=fabric.device, plan=plan, init_data_parallel=False,
    )

    # 5. Fused optimizer via apply_optimizer_in_backward.
    fused_cls, fused_kwargs = _fused_optimizer_from_config(cfg)
    apply_optimizer_in_backward(
        fused_cls, model.parameters(), fused_kwargs,
    )
    # Non-fused params (none in v1 for skip-gram, but plumbed for safety).
    non_fused_params = list(in_backward_optimizer_filter(model.named_parameters()))
    optimizer = (
        torch.optim.SGD([p for _, p in non_fused_params], lr=cfg.learning_rate)
        if non_fused_params
        else _NullOptimizer()
    )

    # 6. Maybe resume.
    start_epoch, start_step = _maybe_resume_checkpoint(
        model, optimizer,
        checkpoint_dir=cfg.checkpoint_dir,
        overwrite=cfg.checkpoint_overwrite,
        current_vocab_path=vocab_path,
    )

    # 7. DataLoader.
    loader = rdf2vec._build_walk_pair_loader(walks_path, vocab_df)
    loader = fabric.setup_dataloaders(loader, use_distributed_sampler=True)

    # 8. Training loop.
    global_step = start_step
    last_ckpt_time = time.monotonic()
    token_counts_dev = token_counts.to(fabric.device)
    for epoch in range(start_epoch, cfg.epochs):
        for batch in loader:
            center, context = batch
            negative = _sample_negatives(
                n_rows=center.size(0), n_neg=cfg.negative_samples,
                vocab_size=vocab_size, device=fabric.device,
                strategy=cfg.negative_sampling, token_counts=token_counts_dev,
            )
            loss = model(center, context, negative)
            fabric.backward(loss)
            # optimizer step is fused into backward via apply_optimizer_in_backward.
            if non_fused_params:
                optimizer.step()
                optimizer.zero_grad()
            rdf2vec.tracker.log({"train/loss": loss.item(), "epoch": epoch, "step": global_step})
            global_step += 1

            # Time-based checkpoint.
            if cfg.checkpoint_dir and (
                time.monotonic() - last_ckpt_time >= cfg.checkpoint_interval_minutes * 60
            ):
                _save_checkpoint(
                    model, optimizer,
                    epoch=epoch, global_step=global_step,
                    checkpoint_dir=cfg.checkpoint_dir,
                    keep_last=cfg.checkpoint_keep_last,
                    vocab_path=vocab_path,
                    config_snapshot=cfg.model_dump(),
                )
                last_ckpt_time = time.monotonic()

        # Epoch-boundary checkpoint.
        if cfg.checkpoint_dir:
            _save_checkpoint(
                model, optimizer,
                epoch=epoch + 1, global_step=global_step,
                checkpoint_dir=cfg.checkpoint_dir,
                keep_last=cfg.checkpoint_keep_last,
                vocab_path=vocab_path,
                config_snapshot=cfg.model_dump(),
            )

    # 9. Stash for transform().
    rdf2vec._torchrec_model = model
    rdf2vec._torchrec_vocab_df = vocab_df


def _fabric_precision_from_config(cfg) -> str:
    # Mirror the matmul_precision knob already in cfg. "high" → "32-true",
    # "medium" → "bf16-mixed" — adjust here if needed at impl time.
    return {
        "highest": "32-true",
        "high":    "32-true",
        "medium":  "bf16-mixed",
    }.get(getattr(cfg, "matmul_precision", "high"), "32-true")


def _fused_optimizer_from_config(cfg) -> tuple[type, dict]:
    """Map cfg.optimizer → (torchrec fused optimizer class, kwargs)."""
    from torchrec.optim.rowwise_adagrad import RowWiseAdagrad  # noqa: F401  # smoke
    if cfg.optimizer == "sgd":
        from torchrec.optim.fused import FusedSGD
        return FusedSGD, {"lr": cfg.learning_rate}
    if cfg.optimizer == "sparseadam":
        from torchrec.optim.fused import PartialRowWiseAdam
        return PartialRowWiseAdam, {"lr": cfg.learning_rate}
    raise ValueError(f"backend='torchrec' does not support optimizer={cfg.optimizer!r}")


class _NullOptimizer:
    """Stand-in when all params are fused via apply_optimizer_in_backward."""
    def step(self): pass
    def zero_grad(self): pass
    def state_dict(self): return {}
    def load_state_dict(self, sd): pass
```

> **Note for the executing engineer:** the exact torchrec optimizer
> import paths (`torchrec.optim.fused.FusedSGD`,
> `torchrec.optim.fused.PartialRowWiseAdam`) may need adjustment to
> match the installed torchrec version. If imports fail, check
> `torchrec.optim.optimizers` or the FBGEMM `EmbOptimType` enum and
> update the two lines in `_fused_optimizer_from_config` accordingly.
> Equivalent name and behaviour are the contract, not the exact import
> path.

- [ ] **Step 2: Add the two `GPU_RDF2Vec` helper methods referenced above**

In `src/rdf2vecgpu/gpu_rdf2vec.py`, add a `_build_vocab_for_walks` method that wraps the existing `_generate_vocab` call to return `(vocab_df, token_counts, vocab_parquet_path)`, and a `_build_walk_pair_loader` method that constructs a `(center, context)` pair dataloader from the walks parquet, using the existing `embedders/word2vec_loader.py` machinery. The exact implementation should mirror what `_fit_via_pytorch_sequences` and `_fit_via_gensim` already do for vocab build and pair generation; factor the relevant block out rather than rewriting it.

> **Note for the executing engineer:** if those helpers don't cleanly factor, prefer inlining the equivalent dask-cudf + token-counts derivation directly in `train_torchrec_word2vec` over rewriting unrelated code. The contract is: `vocab_df: dask_cudf.DataFrame[token_id, entity]`, `token_counts: torch.Tensor[vocab_size]`, `vocab_path: str` (the parquet path written to disk so `_maybe_resume_checkpoint` can verify identity across runs).

- [ ] **Step 3: Run all previously-passing tests to confirm no regression**

Run: `pytest test/embedders/torchrec_word2vec_test.py -v --gpu`
Expected: all Layer 1 + Layer 2 tests still pass; nothing new is exercised yet.

- [ ] **Step 4: Commit**

```bash
git add src/rdf2vecgpu/embedders/torchrec_word2vec.py src/rdf2vecgpu/gpu_rdf2vec.py
git commit -m "feat(torchrec): train_torchrec_word2vec — Fabric loop with DCP checkpoints"
```

---

## Task 12: `transform()` / `fit_transform()` backend-aware return

**Files:**
- Modify: `src/rdf2vecgpu/gpu_rdf2vec.py`

- [ ] **Step 1: Locate the existing `transform` and `fit_transform` methods**

Run: `grep -n "def transform\|def fit_transform" src/rdf2vecgpu/gpu_rdf2vec.py`
Note the line numbers.

- [ ] **Step 2: Add the torchrec branch in `transform`**

Edit `src/rdf2vecgpu/gpu_rdf2vec.py`. At the top of `transform`, before the existing logic, add:

```python
        if self.config.backend == "torchrec":
            if not hasattr(self, "_torchrec_model"):
                raise RuntimeError(
                    "transform() on backend='torchrec' must be called within the "
                    "same distributed launch as fit(). The sharded model is "
                    "rank-local and cannot be reconstructed after the process "
                    "group is destroyed."
                )
            from rdf2vecgpu.embedders.torchrec_word2vec import (
                extract_embeddings_to_dask_cudf,
            )
            return extract_embeddings_to_dask_cudf(
                self._torchrec_model, self._torchrec_vocab_df,
            )
```

- [ ] **Step 3: `fit_transform` already chains to `transform`**

Verify (no edit unless `fit_transform` does its own thing):

Run: `grep -A 5 "def fit_transform" src/rdf2vecgpu/gpu_rdf2vec.py`
Expected: `fit_transform` calls `fit(...)` then `transform(...)` — no change needed.

- [ ] **Step 4: Commit**

```bash
git add src/rdf2vecgpu/gpu_rdf2vec.py
git commit -m "feat(torchrec): transform() returns dask_cudf for backend='torchrec'"
```

---

## Task 13: Multi-GPU integration tests (Layers 3, 4, 5)

These tests run only under `torchrun --standalone --nproc_per_node=2 -m pytest -m multigpu`. They produce no new product code; the failing/passing cycle for them is **run them manually with torchrun before the PR is approved**.

**Files:**
- Modify: `test/embedders/torchrec_word2vec_test.py`

- [ ] **Step 1: Append Layer 3 — end-to-end multi-GPU smoke**

Append to `test/embedders/torchrec_word2vec_test.py`:

```python
# ───────────────────────────── Layer 3 ────────────────────────────────


@pytest.mark.multigpu
def test_torchrec_end_to_end_smoke_world_size_2(tmp_path):
    """Synthetic ~10K-triple graph, world_size=2, 1 epoch.

    Run via: torchrun --standalone --nproc_per_node=2 -m pytest -m multigpu \
        -k test_torchrec_end_to_end_smoke
    """
    import os
    import torch.distributed as dist
    import cudf
    import numpy as np

    if not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("requires torchrun --nproc_per_node>=2")

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    rng = np.random.default_rng(0)
    n = 10_000
    triples = cudf.DataFrame({
        "subject":   rng.integers(0, 500, size=n),
        "predicate": rng.integers(0, 20, size=n),
        "object":    rng.integers(500, 1000, size=n),
    })

    cfg = RDF2VecConfig(
        backend="torchrec",
        embedding_model="skipgram",
        multi_gpu=True,
        walk_strategy="random",
        walk_depth=3, walk_number=5,
        vector_size=8, epochs=1,
        learning_rate=0.025, negative_samples=5,
        optimizer="sparseadam",
    )
    model = GPU_RDF2Vec(config=cfg)
    model.fit(edge_df=triples)
    embeddings = model.transform()

    # Each rank produces one partition.
    assert embeddings.npartitions == dist.get_world_size()

    # Materialised row count covers the full vocab exactly once.
    pdf = embeddings.compute().to_pandas()
    assert len(pdf) == pdf["token_id"].nunique()
```

- [ ] **Step 2: Append Layer 4 — cross-backend equivalence**

Append:

```python
# ───────────────────────────── Layer 4 ────────────────────────────────


@pytest.mark.multigpu
def test_torchrec_vs_pytorch_topk_overlap(tmp_path):
    """Quality canary: top-K NN overlap > 70 % between pytorch and torchrec.

    Not bit-exact — PARTIAL_ROWWISE_ADAM ≠ SparseAdam. Catches large
    regressions (broken sharding, wrong sign of loss, etc.).
    """
    import os
    import torch.distributed as dist
    import cudf
    import numpy as np
    import pandas as pd

    if not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("requires torchrun --nproc_per_node>=2")

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    rng = np.random.default_rng(42)
    n = 5_000
    triples = cudf.DataFrame({
        "subject":   rng.integers(0, 200, size=n),
        "predicate": rng.integers(0, 10, size=n),
        "object":    rng.integers(200, 400, size=n),
    })

    common = dict(
        embedding_model="skipgram", walk_strategy="random",
        walk_depth=3, walk_number=10,
        vector_size=16, epochs=2,
        learning_rate=0.025, negative_samples=5,
        random_state=0,
    )

    cfg_pt = RDF2VecConfig(backend="pytorch", multi_gpu=False, **common)
    emb_pt = GPU_RDF2Vec(config=cfg_pt).fit_transform(edge_df=triples).to_pandas()

    cfg_tr = RDF2VecConfig(backend="torchrec", multi_gpu=True, **common)
    emb_tr = GPU_RDF2Vec(config=cfg_tr).fit_transform(edge_df=triples).compute().to_pandas()

    # Compute top-5 NN overlap per token.
    def topk_neighbours(df: pd.DataFrame, k: int = 5) -> dict[int, set[int]]:
        m = df.set_index("token_id").drop(columns=["entity"]).to_numpy()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        cos = (m @ m.T) / (norms * norms.T + 1e-12)
        np.fill_diagonal(cos, -np.inf)
        return {
            int(tid): set(int(t) for t in df["token_id"].iloc[np.argsort(-cos[i])[:k]])
            for i, tid in enumerate(df["token_id"].tolist())
        }

    nn_pt = topk_neighbours(emb_pt)
    nn_tr = topk_neighbours(emb_tr)
    overlap = np.mean([len(nn_pt[t] & nn_tr.get(t, set())) / 5 for t in nn_pt])
    assert overlap > 0.7, f"top-5 NN overlap too low: {overlap:.2f}"
```

- [ ] **Step 3: Append Layer 5 — checkpoint round-trip + auto-resume + resharding + overwrite + vocab mismatch + keep_last**

Append:

```python
# ───────────────────────────── Layer 5 ────────────────────────────────


@pytest.mark.multigpu
def test_save_then_reload_preserves_embeddings_bit_equal(tmp_path):
    """Round trip: train, extract, simulate restart, reload, extract again.

    Bit-equal between the two extractions verifies that DCP saved and
    loaded the sharded weights without drift.
    """
    import torch.distributed as dist
    import cudf
    import numpy as np

    if not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("requires torchrun --nproc_per_node>=2")

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    rng = np.random.default_rng(7)
    triples = cudf.DataFrame({
        "subject":   rng.integers(0, 100, size=2000),
        "predicate": rng.integers(0, 5,   size=2000),
        "object":    rng.integers(100, 200, size=2000),
    })

    ckpt = tmp_path / "ckpt"
    cfg = RDF2VecConfig(
        backend="torchrec", embedding_model="skipgram", multi_gpu=True,
        walk_strategy="random", walk_depth=3, walk_number=5,
        vector_size=8, epochs=2,
        learning_rate=0.025, negative_samples=5,
        checkpoint_dir=str(ckpt), checkpoint_interval_minutes=120,
    )

    # First run: train, capture embeddings.
    m1 = GPU_RDF2Vec(config=cfg)
    emb1 = m1.fit_transform(edge_df=triples).compute().to_pandas().sort_values("token_id")

    # Simulate restart: build a new model with epochs=2 and the same
    # checkpoint_dir → auto-resume from epoch 2 (no further training).
    m2 = GPU_RDF2Vec(config=cfg)
    m2.fit(edge_df=triples)
    emb2 = m2.transform().compute().to_pandas().sort_values("token_id")

    np.testing.assert_array_equal(
        emb1.drop(columns=["entity"]).to_numpy(),
        emb2.drop(columns=["entity"]).to_numpy(),
    )


@pytest.mark.multigpu
def test_overwrite_starts_fresh(tmp_path):
    import torch.distributed as dist
    import cudf
    import numpy as np

    if not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("requires torchrun --nproc_per_node>=2")

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    rng = np.random.default_rng(11)
    triples = cudf.DataFrame({
        "subject":   rng.integers(0, 50, size=1000),
        "predicate": rng.integers(0, 5,  size=1000),
        "object":    rng.integers(50, 100, size=1000),
    })

    ckpt = tmp_path / "ckpt"

    cfg = RDF2VecConfig(
        backend="torchrec", embedding_model="skipgram", multi_gpu=True,
        walk_strategy="random", walk_depth=3, walk_number=5,
        vector_size=8, epochs=1,
        checkpoint_dir=str(ckpt),
    )
    GPU_RDF2Vec(config=cfg).fit(edge_df=triples)
    # Verify a checkpoint was actually written
    assert (ckpt / "latest").exists()

    cfg_ow = cfg.model_copy(update={"checkpoint_overwrite": True})
    GPU_RDF2Vec(config=cfg_ow).fit(edge_df=triples)
    # After overwrite + fresh fit, exactly one step_* dir again
    step_dirs = [p for p in ckpt.iterdir() if p.name.startswith("step_")]
    assert len(step_dirs) >= 1


@pytest.mark.multigpu
def test_keep_last_prunes_excess_step_dirs(tmp_path):
    """Drive enough time-based saves to test pruning end-to-end."""
    import torch.distributed as dist
    import cudf
    import numpy as np

    if not dist.is_initialized() or dist.get_world_size() < 2:
        pytest.skip("requires torchrun --nproc_per_node>=2")

    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    rng = np.random.default_rng(13)
    triples = cudf.DataFrame({
        "subject":   rng.integers(0, 100, size=5000),
        "predicate": rng.integers(0, 5,   size=5000),
        "object":    rng.integers(100, 200, size=5000),
    })
    ckpt = tmp_path / "ckpt"
    cfg = RDF2VecConfig(
        backend="torchrec", embedding_model="skipgram", multi_gpu=True,
        walk_strategy="random", walk_depth=3, walk_number=20,
        vector_size=8, epochs=5,
        checkpoint_dir=str(ckpt),
        checkpoint_interval_minutes=1,   # force frequent saves
        checkpoint_keep_last=2,
    )
    GPU_RDF2Vec(config=cfg).fit(edge_df=triples)
    step_dirs = sorted(p.name for p in ckpt.iterdir() if p.name.startswith("step_"))
    assert len(step_dirs) == 2, step_dirs
```

> **Note:** the resharding test (save at world_size=2, reload at
> world_size=1) requires spawning two separate `torchrun` invocations
> from the test harness. That is awkward to express as a single pytest
> case; document the manual procedure in `README.md` (Task 14) and skip
> the automated version in v1.

- [ ] **Step 4: Run the multi-GPU suite locally**

Run: `torchrun --standalone --nproc_per_node=2 -m pytest test/embedders/torchrec_word2vec_test.py -m multigpu -v`
Expected: all multigpu-marked tests pass on a 2-GPU box. On CI (1 GPU), they skip.

- [ ] **Step 5: Commit**

```bash
git add test/embedders/torchrec_word2vec_test.py
git commit -m "test(torchrec): Layer 3/4/5 multi-GPU integration + checkpoint tests"
```

---

## Task 14: README + docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the install snippet to the "Repository setup" section**

After the existing `cudf` install snippet in README.md, add:

```markdown
3. (Optional) Install the torchrec extra for the sharded-embedding backend:

```bash
pip install "rdf2vecgpu[torchrec]"
```

The torchrec backend is what you want when the in/out embedding tables
no longer fit in a single GPU's VRAM. See
[Sharded embeddings (torchrec backend)](#sharded-embeddings-torchrec-backend)
below.
```

- [ ] **Step 2: Add the new subsection under "Implementation Details"**

After the existing "Distributed correctness — when to `persist`"
subsection in README.md, add:

```markdown
6. **Sharded embeddings (torchrec backend)**

   For graphs whose in/out embedding tables exceed a single GPU's VRAM,
   set `backend="torchrec"` on `RDF2VecConfig`. This swaps in:

   - **Row-wise sharded embedding tables** across all ranks via
     [torchrec](https://github.com/pytorch/torchrec). Per-rank VRAM
     footprint scales as `total_table_bytes / world_size`.
   - **Fused embedding+optimizer kernels** via FBGEMM
     (`PARTIAL_ROWWISE_ADAM` for `optimizer="sparseadam"`,
     `EXACT_SGD` for `optimizer="sgd"`). Note: `PARTIAL_ROWWISE_ADAM`
     keeps per-row moment state, not per-cell like
     `torch.optim.SparseAdam` — close but not bit-identical.
   - **Lightning Fabric** runner in place of `pl.Trainer`. Launch via
     `torchrun`:

     ```bash
     torchrun --standalone --nproc_per_node=2 your_script.py
     ```

   - **`dask_cudf.DataFrame` return type** from `fit_transform` /
     `transform` (one partition per rank). Call
     `embeddings.to_parquet("dir/")` to write a partitioned dataset.
   - **DCP-based checkpointing** when `checkpoint_dir` is set. Saves on
     each epoch boundary and every `checkpoint_interval_minutes`.
     Auto-resumes if `checkpoint_dir` contains a `latest` symlink. Pass
     `checkpoint_overwrite=True` to discard and start fresh.

   v1 covers skip-gram only; CBOW and order-aware skip-gram remain on
   the existing `backend="pytorch"` path.

   **Multi-node note.** Checkpoints rely on a filesystem path that
   resolves to the same physical storage on every rank. Single-node
   multi-GPU is fully supported as-is; multi-node requires a shared
   filesystem (NFS, Lustre, S3 via fsspec, etc.) at the same path.

   **Resharding.** A checkpoint saved at `world_size=N` can resume at
   `world_size=M ≠ N` — DCP redistributes the saved shards to match
   the current plan. The vocab itself must be unchanged across the
   resume; we guard against accidental vocab drift via a `vocab_path`
   recorded in `metadata.pt`.
```

- [ ] **Step 3: Add an entry under "Roadmap"**

Replace the existing roadmap entry list's first unchecked item with:

```markdown
- [x] Sharded embedding tables via torchrec for graphs whose vocabulary
  exceeds single-GPU VRAM (`backend="torchrec"`, v0.5.0).
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): sharded-embeddings (torchrec backend) install + usage"
```

---

## Self-review against spec

(Run mentally before declaring the plan complete.)

**Spec coverage:**
- §2 architecture overview — Task 1 (pyproject extras), Task 2 (config), Task 3 (dispatch), Tasks 4–11 (module), Task 12 (transform), Task 14 (README). ✓
- §3 components — every listed function/method has a task: `ShardedSkipGramModel` (4, 6), `_build_sharding_plan` (7), `_sample_negatives` (5), `train_torchrec_word2vec` (11), `extract_embeddings_to_dask_cudf` (8), `_save_checkpoint` (9), `_maybe_resume_checkpoint` (10), config fields (2), dispatch (3, 12), pyproject (1), README (14). ✓
- §4 data flow — Task 11 implements all 10 steps; checkpoint subsection covered by Tasks 9, 10 and exercised by Task 13. ✓
- §5 error handling — pre-flight in Task 3; checkpoint-specific guards in Task 10; transform guard in Task 12; sharding-plan OOM and rank desync are framework-level (no dedicated task). ✓
- §6 testing — Layer 1 in Task 3, Layer 2 in Tasks 4–10, Layers 3/4/5 in Task 13. ✓

**Placeholder scan:** two notes-for-engineer remain in Task 11 (optimizer import paths; vocab helper factoring). These are not placeholders in the prohibited sense — they're explicit contracts with a defined fallback. Acceptable.

**Type consistency:** `extract_embeddings_to_dask_cudf` signature (`model`, `vocab_df`) matches both Task 8 (definition) and Task 12 (call site). `_save_checkpoint` / `_maybe_resume_checkpoint` signatures consistent across Tasks 9, 10, 11. `IN_TABLE_NAME` / `OUT_TABLE_NAME` constants on `ShardedSkipGramModel` referenced consistently in Tasks 4, 7, 8.

No gaps found.
