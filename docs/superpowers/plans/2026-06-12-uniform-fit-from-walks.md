# Uniform `fit_from_walks` + GPU-resident single-GPU feed — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose one decoupled `prepare_corpus` → `fit_from_walks(walks_path, vocab_path)` seam across all four backends (gensim, pytorch single-GPU, pytorch multi-GPU, torchrec), and restore the in-memory GPU-resident data feed as the single-GPU pytorch default.

**Architecture:** A single canonical on-disk corpus (walks-as-sequences parquet + `vocab.parquet`, already produced by `prepare_corpus`) feeds every backend. `fit_from_walks` becomes backend-dispatched. Single-GPU pytorch trains from an in-memory cuDF frame via a new DLPack-based `InMemoryWalkSeqDataModule` (no temp parquet, no CPU decode); multi-GPU streams the parquet per rank under DDP; gensim and torchrec keep their existing training functions.

**Tech Stack:** cuDF / cuGraph (RAPIDS 25.06), PyTorch + Lightning, DLPack zero-copy, gensim, torchrec (cu128), pytest (`gpu` / `multigpu` markers).

**Design spec:** `docs/superpowers/specs/2026-06-12-uniform-fit-from-walks-design.md`

**Environment:** conda env `rdf2vecgpu_test`. Single-GPU tests: `pytest --gpu`. Multi-GPU: `NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 -m pytest -m multigpu`. The `--gpu` marker gate lives in `test/conftest.py`.

---

## File Structure

- **`src/rdf2vecgpu/embedders/word2vec_loader.py`** — add `InMemoryWalkSeqDataModule` (+ helper `_InMemoryWalkSeq`): GPU-native sibling of `WalkSequenceDataModule`.
- **`src/rdf2vecgpu/gpu_rdf2vec.py`** — add `_token_counts_from_cudf_walks`, `_load_word2idx`, `_train_pytorch_sequences`; rework the single-GPU branch of `_fit_via_pytorch_sequences`; generalise `fit_from_walks` from torchrec-only to backend dispatch.
- **`test/embedders/word2vec_loader_test.py`** — tests for `InMemoryWalkSeqDataModule`.
- **`test/embedders/fit_from_walks_test.py`** — new: `_load_word2idx`, `_token_counts_from_cudf_walks`, dispatch + integration (all four backends from a shared corpus).
- **`performance/bench_backends.py`** — switch to `prepare_corpus` once + `fit_from_walks` only.
- **`README.md` / `CHANGELOG.md`** — document `fit_from_walks` as the cross-backend decoupled entry.

---

## Task 1: `InMemoryWalkSeqDataModule` — GPU-resident in-memory loader

**Files:**
- Modify: `src/rdf2vecgpu/embedders/word2vec_loader.py` (add at end of file)
- Test: `test/embedders/word2vec_loader_test.py`

- [ ] **Step 1: Write the failing test** (append to the test file)

```python
import pytest


@pytest.mark.gpu
def test_inmemory_walkseq_yields_gpu_tensors_of_right_shape():
    import cudf
    import torch
    from rdf2vecgpu.embedders.word2vec_loader import InMemoryWalkSeqDataModule

    # 5 walks, depth=3 → vertices length 4, predicates length 3 (cuGraph -1 pads).
    df = cudf.DataFrame({
        "vertices":   [[0, 1, 2, 3], [1, 2, 3, -1], [2, 0, -1, -1],
                       [3, 1, 0, 2], [0, 0, 1, -1]],
        "predicates": [[10, 11, 12], [11, 12, -1], [12, -1, -1],
                       [10, 10, 11], [11, 12, -1]],
    })
    dm = InMemoryWalkSeqDataModule(df, emit_batch_size=2, shuffle=False)
    dm.setup()
    batches = list(dm.train_dataloader())

    # 5 rows, emit_batch_size=2 → 3 batches (2, 2, 1).
    assert [b[0].shape[0] for b in batches] == [2, 2, 1]
    v0, p0 = batches[0]
    assert v0.is_cuda and p0.is_cuda            # GPU-resident, no CPU round-trip
    assert v0.shape == (2, 4) and p0.shape == (2, 3)
    # Round-trips the data unchanged (shuffle=False).
    assert v0[0].tolist() == [0, 1, 2, 3]
    assert p0[0].tolist() == [10, 11, 12]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest --gpu test/embedders/word2vec_loader_test.py::test_inmemory_walkseq_yields_gpu_tensors_of_right_shape -v`
Expected: FAIL — `ImportError: cannot import name 'InMemoryWalkSeqDataModule'`.

- [ ] **Step 3: Implement `InMemoryWalkSeqDataModule`** (append to `word2vec_loader.py`)

```python
class _InMemoryWalkSeq(IterableDataset):
    """GPU-resident sibling of `_WalkSeqIterable`. Materialises the
    walks-as-sequences cuDF list columns to 2-D GPU int tensors ONCE (via
    DLPack, zero-copy off the device), then yields `emit_batch_size`-row
    `(vertices, predicates)` GPU slices. Single-GPU only — the whole corpus
    lives in one device's memory, so there is no parquet/CPU hop."""

    def __init__(self, cudf_df, *, emit_batch_size, shuffle=True, seed=0):
        super().__init__()
        for col in ("vertices", "predicates"):
            if col not in cudf_df.columns:
                raise ValueError(f"in-memory walks frame missing '{col}' list column")
        n = len(cudf_df)
        # cuDF list column -> flat leaves Series -> GPU tensor -> (n, row_len).
        # All walks share a fixed length (cuGraph pads short walks with -1), so
        # the leaf count is exactly n * row_len and the view is well-defined.
        self.vertices = torch.from_dlpack(
            cudf_df["vertices"].list.leaves.to_dlpack()
        ).view(n, -1) if n else torch.empty((0, 0), dtype=torch.int32, device="cuda")
        self.predicates = torch.from_dlpack(
            cudf_df["predicates"].list.leaves.to_dlpack()
        ).view(n, -1) if n else torch.empty((0, 0), dtype=torch.int32, device="cuda")
        self.emit_batch_size = int(emit_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = -1

    def __iter__(self):
        self._epoch += 1
        n = self.vertices.shape[0]
        if self.shuffle and n:
            g = torch.Generator(device=self.vertices.device)
            g.manual_seed(self.seed + self._epoch)
            perm = torch.randperm(n, generator=g, device=self.vertices.device)
            v, p = self.vertices[perm], self.predicates[perm]
        else:
            v, p = self.vertices, self.predicates
        eb = self.emit_batch_size
        for s in range(0, n, eb):
            yield v[s:s + eb], p[s:s + eb]


class InMemoryWalkSeqDataModule(L.LightningDataModule):
    """Feeds `(vertices, predicates)` GPU int tensors straight from an in-memory
    cuDF walks-as-sequences frame — the GPU-resident counterpart of
    `WalkSequenceDataModule`. num_workers is forced to 0: the batches are CUDA
    tensors, which cannot cross DataLoader worker processes."""

    def __init__(self, cudf_df, *, emit_batch_size, shuffle=True, seed=0):
        super().__init__()
        self.cudf_df = cudf_df
        self.emit_batch_size = emit_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._iterable = None

    def setup(self, stage=None):
        self._iterable = _InMemoryWalkSeq(
            self.cudf_df,
            emit_batch_size=self.emit_batch_size,
            shuffle=self.shuffle,
            seed=self.seed,
        )

    def train_dataloader(self):
        if self._iterable is None:
            self.setup()
        return DataLoader(
            self._iterable,
            batch_size=None,        # the iterable yields already-batched GPU tensors
            num_workers=0,          # CUDA tensors can't cross worker processes
            pin_memory=False,       # already on GPU
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest --gpu test/embedders/word2vec_loader_test.py::test_inmemory_walkseq_yields_gpu_tensors_of_right_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/embedders/word2vec_loader.py test/embedders/word2vec_loader_test.py
git commit -m "feat(loader): InMemoryWalkSeqDataModule — GPU-resident walks feed via DLPack"
```

---

## Task 2: `_token_counts_from_cudf_walks` — GPU token counts for the in-memory path

The single-GPU in-memory path has no parquet, so it can't use `_token_counts_from_walk_parquet`. Add a cuDF/cupy equivalent.

**Files:**
- Modify: `src/rdf2vecgpu/gpu_rdf2vec.py` (add a module-level function next to `_token_counts_from_walk_parquet`)
- Test: `test/embedders/fit_from_walks_test.py` (new file)

- [ ] **Step 1: Write the failing test** (create the test file)

```python
import pytest


@pytest.mark.gpu
def test_token_counts_from_cudf_walks_skips_sentinels():
    import cudf
    from rdf2vecgpu.gpu_rdf2vec import _token_counts_from_cudf_walks

    df = cudf.DataFrame({
        "vertices":   [[0, 1, 2, -1], [1, 1, -1, -1]],
        "predicates": [[3, 3, -1], [3, -1, -1]],
    })
    counts = _token_counts_from_cudf_walks(df, vocab_size=4)
    # token 0 →1, token 1 →3 (two in vertices + one), token 2 →1, token 3 →3; -1 skipped.
    assert counts.tolist() == [1, 3, 1, 3]
    assert counts.dtype.name == "int64"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest --gpu test/embedders/fit_from_walks_test.py::test_token_counts_from_cudf_walks_skips_sentinels -v`
Expected: FAIL — `ImportError: cannot import name '_token_counts_from_cudf_walks'`.

- [ ] **Step 3: Implement** (add near `_token_counts_from_walk_parquet` in `gpu_rdf2vec.py`)

```python
def _token_counts_from_cudf_walks(cudf_df, vocab_size: int):
    """Token occurrence counts over an in-memory walks-as-sequences cuDF frame
    → numpy int64 array of length ``vocab_size``. GPU equivalent of
    ``_token_counts_from_walk_parquet`` for the single-GPU in-memory path:
    flattens the ``vertices`` / ``predicates`` list columns on the device and
    bincounts, dropping the ``-1`` cuGraph walk sentinels."""
    import cupy
    import numpy as np

    counts = cupy.zeros(vocab_size, dtype=cupy.int64)
    for col in ("vertices", "predicates"):
        leaves = cupy.asarray(cudf_df[col].list.leaves)
        leaves = leaves[leaves >= 0]
        if leaves.size:
            counts += cupy.bincount(leaves, minlength=vocab_size)[:vocab_size]
    return cupy.asnumpy(counts).astype(np.int64)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest --gpu test/embedders/fit_from_walks_test.py::test_token_counts_from_cudf_walks_skips_sentinels -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/gpu_rdf2vec.py test/embedders/fit_from_walks_test.py
git commit -m "feat: _token_counts_from_cudf_walks — GPU token counts for the in-memory path"
```

---

## Task 3: `_load_word2idx` — load the prepared vocab into `(token, word)`

**Files:**
- Modify: `src/rdf2vecgpu/gpu_rdf2vec.py` (new method on `GPU_RDF2Vec`)
- Test: `test/embedders/fit_from_walks_test.py`

- [ ] **Step 1: Write the failing test** (append)

```python
@pytest.mark.gpu
def test_load_word2idx_renames_corpus_vocab(tmp_path):
    import cudf
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    vocab_path = str(tmp_path / "vocab.parquet")
    cudf.DataFrame({"token_id": [0, 1, 2], "entity": ["a", "b", "c"]}).to_parquet(vocab_path)

    m = GPU_RDF2Vec(config=RDF2VecConfig(backend="pytorch", embedding_model="skipgram"))
    out = m._load_word2idx(vocab_path)

    assert list(out.columns) == ["token", "word"]
    assert m.word2idx is out
    assert out["token"].to_arrow().to_pylist() == [0, 1, 2]
    assert out["word"].to_arrow().to_pylist() == ["a", "b", "c"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest --gpu test/embedders/fit_from_walks_test.py::test_load_word2idx_renames_corpus_vocab -v`
Expected: FAIL — `AttributeError: 'GPU_RDF2Vec' object has no attribute '_load_word2idx'`.

- [ ] **Step 3: Implement** (add method to `GPU_RDF2Vec`)

```python
    def _load_word2idx(self, vocab_path: str):
        """Load a prepared corpus vocab (`token_id`, `entity` — as written by
        `prepare_corpus`) into `self.word2idx` in the `(token, word)` layout the
        pytorch / gensim extract paths expect, and return it. The walks were
        encoded against these same token ids, so the mapping is consistent by
        construction."""
        vocab = cudf.read_parquet(vocab_path)
        self.word2idx = vocab.rename(columns={"token_id": "token", "entity": "word"})[
            ["token", "word"]
        ]
        return self.word2idx
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest --gpu test/embedders/fit_from_walks_test.py::test_load_word2idx_renames_corpus_vocab -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/gpu_rdf2vec.py test/embedders/fit_from_walks_test.py
git commit -m "feat: _load_word2idx — load prepared corpus vocab for pytorch/gensim"
```

---

## Task 4: Extract `_train_pytorch_sequences` (refactor, no behaviour change)

`_fit_via_pytorch_sequences` currently builds the `SkipGram` model, the `L.Trainer`, and calls `fit` inline. Extract that tail into a reusable helper so `fit_from_walks` (Task 6) and the in-memory branch (Task 5) share it. This is a pure refactor — existing single-GPU pytorch behaviour is unchanged.

**Files:**
- Modify: `src/rdf2vecgpu/gpu_rdf2vec.py` — `_fit_via_pytorch_sequences` and a new `_train_pytorch_sequences`

- [ ] **Step 1: Confirm the baseline single-GPU pytorch test passes** (regression guard for the refactor)

Run: `pytest --gpu -k "pytorch and skipgram" test/embedders/word2vec_test.py -v`
Expected: PASS (note which tests run; these are the guard).

- [ ] **Step 2: Add `_train_pytorch_sequences`** (new method on `GPU_RDF2Vec`)

Move the model-construction + Trainer + fit block out of `_fit_via_pytorch_sequences` into this method. Signature and body:

```python
    def _train_pytorch_sequences(self, datamodule, *, token_counts, steps_per_epoch, n_gpus):
        """Build the SkipGram model + Lightning Trainer and fit on `datamodule`.
        Shared by `_fit_via_pytorch_sequences` (fit()) and `fit_from_walks`.
        `datamodule` yields `(vertices, predicates)` batches — either
        `InMemoryWalkSeqDataModule` (single-GPU, GPU-resident) or
        `WalkSequenceDataModule` (multi-GPU, parquet streamed per rank)."""
        torch.set_float32_matmul_precision(self.config.matmul_precision)
        self._walked_mask = (token_counts > 0)
        model = SkipGram(
            vocab_size=self.word2idx.shape[0],
            embedding_dim=self.config.vector_size,
            neg_samples=self.config.negative_samples,
            learning_rate=self.config.learning_rate,
            use_sparse=True,
            epochs=self.config.epochs,
            min_learning_rate=self.config.min_learning_rate,
            lr_decay=self.config.lr_decay,
            window=self.config.window_size,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            steps_per_epoch=steps_per_epoch,
            grad_clip_value=self.config.grad_clip_value,
            optimizer=self.config.optimizer,
            grad_clip_mode=self.config.grad_clip_mode,
            sample=self.config.sample,
            negative_sampling=self.config.negative_sampling,
            token_counts=token_counts,
        )
        if self.config.reproducible:
            L.seed_everything(self.config.random_state, workers=True)
        self.tracker.log_pytorch()
        trainer = L.Trainer(
            max_epochs=self.config.epochs,
            log_every_n_steps=1,
            accelerator="gpu",
            precision=self.config.matmul_precision_trainer if hasattr(
                self.config, "matmul_precision_trainer") else "32-true",
            devices=n_gpus,
            strategy="ddp" if n_gpus > 1 else "auto",
            num_nodes=self.config.num_nodes,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            gradient_clip_val=None,   # SkipGram clips per-row/element internally
            enable_checkpointing=False,
        )
        trainer.fit(model, datamodule)
        self.tracker.log_model_pytorch(model, artifact_path="word2vec_model")
        self.word2vec_model = model
```

> NOTE for the implementer: copy the EXACT `SkipGram(...)` kwargs and `L.Trainer(...)` kwargs currently present in `_fit_via_pytorch_sequences` (around lines 1330–1360) rather than the illustrative versions above — match the live code so behaviour is byte-identical. The signature (`datamodule`, `token_counts`, `steps_per_epoch`, `n_gpus`) is the contract.

- [ ] **Step 3: Rewrite `_fit_via_pytorch_sequences` to call the helper**

Keep the existing temp-parquet write + token-count computation, then replace the inline model/Trainer block with:

```python
        datamodule = WalkSequenceDataModule(
            walks_parquet, batch_size=walks_per_batch,
            num_workers=self.config.cpu_count, shuffle=True, seed=self.config.random_state,
        )
        self._train_pytorch_sequences(
            datamodule, token_counts=token_counts,
            steps_per_epoch=steps_per_epoch, n_gpus=n_gpus,
        )
```

- [ ] **Step 4: Run the regression guard**

Run: `pytest --gpu -k "pytorch and skipgram" test/embedders/word2vec_test.py -v`
Expected: PASS (same tests as Step 1, still green).

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/gpu_rdf2vec.py
git commit -m "refactor: extract _train_pytorch_sequences from _fit_via_pytorch_sequences"
```

---

## Task 5: Single-GPU `_fit_via_pytorch_sequences` → in-memory GPU feed

Restore the pre-spike default: single-GPU skip-gram trains from the in-memory cuDF walks via `InMemoryWalkSeqDataModule` — no temp parquet, no CPU decode. Multi-GPU keeps the temp-parquet + DDP path.

**Files:**
- Modify: `src/rdf2vecgpu/gpu_rdf2vec.py` — `_fit_via_pytorch_sequences`
- Test: `test/embedders/fit_from_walks_test.py`

- [ ] **Step 1: Write the failing test** (append)

```python
@pytest.mark.gpu
def test_single_gpu_fit_uses_in_memory_feed_no_temp_parquet(tmp_path, monkeypatch):
    """Single-GPU skip-gram fit() must NOT round-trip walks through temp parquet;
    it uses the GPU-resident InMemoryWalkSeqDataModule."""
    import cudf
    import rdf2vecgpu.gpu_rdf2vec as g
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    # Fail loudly if the single-GPU path writes a temp parquet dir.
    monkeypatch.setattr(g.tempfile, "mkdtemp",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("temp parquet used")))

    triples = str(tmp_path / "t.parquet")
    cudf.DataFrame({
        "subject": [f"e{i%40}" for i in range(2000)],
        "predicate": [f"p{i%5}" for i in range(2000)],
        "object": [f"e{(i*7)%40}" for i in range(2000)],
    }).to_parquet(triples)

    cfg = RDF2VecConfig(backend="pytorch", embedding_model="skipgram",
                        walk_strategy="random", walk_depth=3, walk_number=5,
                        vector_size=8, epochs=1, multi_gpu=False)
    m = GPU_RDF2Vec(config=cfg)
    edges = m.load_data(triples)
    m.fit(edges)                       # must not raise (no mkdtemp)
    emb = m.transform().to_pandas()
    import numpy as np
    vec = [c for c in emb.columns if c not in ("token", "word")]
    assert np.isfinite(emb[vec].to_numpy()).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest --gpu test/embedders/fit_from_walks_test.py::test_single_gpu_fit_uses_in_memory_feed_no_temp_parquet -v`
Expected: FAIL — `AssertionError: temp parquet used` (current path still writes temp parquet).

- [ ] **Step 3: Rework `_fit_via_pytorch_sequences`** to branch on mode

At the top of the method, branch single-GPU vs multi-GPU. The single-GPU branch trains from memory:

```python
        from .embedders.word2vec_loader import (
            InMemoryWalkSeqDataModule, WalkSequenceDataModule, walks_per_pair_batch,
        )
        torch.set_float32_matmul_precision(self.config.matmul_precision)

        # word2idx may be dask_cudf in multi-GPU mode → materialise for vocab size
        if hasattr(self.word2idx, "compute"):
            self.word2idx = self.word2idx.compute()

        walks_per_batch = walks_per_pair_batch(
            self.config.pair_batch_size, self.config.walk_depth, self.config.window_size
        )

        if not self.config.multi_gpu and not hasattr(walk_sequences, "persist"):
            # ── Single-GPU: GPU-resident in-memory feed (no parquet, no CPU) ──
            if hasattr(walk_sequences, "compute"):
                walk_sequences = walk_sequences.compute()
            n_walks = int(len(walk_sequences))
            steps_per_epoch = max(1, -(-n_walks // walks_per_batch))
            token_counts = _token_counts_from_cudf_walks(
                walk_sequences, self.word2idx.shape[0]
            )
            dm = InMemoryWalkSeqDataModule(
                walk_sequences, emit_batch_size=walks_per_batch,
                shuffle=True, seed=self.config.random_state,
            )
            with self.tracker.stage("Word2Vec_Training"):
                self._train_pytorch_sequences(
                    dm, token_counts=token_counts,
                    steps_per_epoch=steps_per_epoch, n_gpus=1,
                )
            return

        # ── Multi-GPU: checkpoint to parquet, tear down dask, DDP-train ──
        # (existing body: temp parquet write, Comms.destroy, client.close,
        #  _token_counts_from_walk_parquet, WalkSequenceDataModule,
        #  self._train_pytorch_sequences(..., n_gpus=torch.cuda.device_count()))
```

> NOTE for the implementer: keep the existing multi-GPU body verbatim below the single-GPU early-return; only the single-GPU branch is new. Ensure `_token_counts_from_cudf_walks` is imported/visible (it is a module-level function in `gpu_rdf2vec.py`).

- [ ] **Step 4: Run the new test + the regression guard**

Run: `pytest --gpu test/embedders/fit_from_walks_test.py::test_single_gpu_fit_uses_in_memory_feed_no_temp_parquet -v`
Expected: PASS.
Run: `pytest --gpu -k "pytorch and skipgram" test/embedders/word2vec_test.py -v`
Expected: PASS (single-GPU training still correct).

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/gpu_rdf2vec.py test/embedders/fit_from_walks_test.py
git commit -m "feat: single-GPU pytorch fit() trains from in-memory cuDF (no temp parquet)"
```

---

## Task 6: Generalise `fit_from_walks` to all backends

**Files:**
- Modify: `src/rdf2vecgpu/gpu_rdf2vec.py` — `fit_from_walks`
- Test: `test/embedders/fit_from_walks_test.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_fit_from_walks_missing_corpus_raises(tmp_path):
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec
    m = GPU_RDF2Vec(config=RDF2VecConfig(backend="gensim", embedding_model="skipgram"))
    with pytest.raises(FileNotFoundError):
        m.fit_from_walks(str(tmp_path / "missing_walks"), str(tmp_path / "missing.parquet"))


def test_fit_from_walks_dispatches_per_backend(tmp_path, monkeypatch):
    import cudf
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    walks = tmp_path / "walks"; walks.mkdir()
    (walks / "part.0.parquet").write_bytes(b"x")          # existence only
    vocab = str(tmp_path / "vocab.parquet")
    cudf.DataFrame({"token_id": [0, 1], "entity": ["a", "b"]}).to_parquet(vocab)

    called = {}
    m = GPU_RDF2Vec(config=RDF2VecConfig(backend="gensim", embedding_model="skipgram"))
    monkeypatch.setattr(m, "_load_word2idx", lambda vp: called.setdefault("vocab", vp))
    monkeypatch.setattr("rdf2vecgpu.gpu_rdf2vec.train_gensim_word2vec",
                        lambda **kw: called.setdefault("gensim", kw) or "model")
    m.fit_from_walks(str(walks), vocab)
    assert called["gensim"]["walks_dir"] == str(walks)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest test/embedders/fit_from_walks_test.py -k "missing_corpus or dispatches" -v`
Expected: FAIL — `fit_from_walks` is torchrec-only / `train_gensim_word2vec` not importable at that path.

- [ ] **Step 3: Implement the dispatch**

Add the import at the top of `gpu_rdf2vec.py`:
```python
from .embedders.gensim_word2vec import train_gensim_word2vec, load_id_to_word
```
Replace the body of `fit_from_walks` (keep the docstring; expand it to mention all backends):
```python
        import os
        if not os.path.exists(walks_path):
            raise FileNotFoundError(f"walks_path does not exist: {walks_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab_path does not exist: {vocab_path}")

        backend = self.config.backend
        if backend == "torchrec":
            self._torchrec_preflight()
            from rdf2vecgpu.embedders.torchrec_word2vec import train_torchrec_word2vec
            train_torchrec_word2vec(self, walks_path, vocab_path)
            return

        self._load_word2idx(vocab_path)

        if backend == "gensim":
            with self.tracker.stage("Word2Vec_Training"):
                self.word2vec_model = train_gensim_word2vec(
                    walks_dir=walks_path, word2idx_df=self.word2idx,
                    vector_size=self.config.vector_size, window=self.config.window_size,
                    min_count=self.config.min_count, embedding_model=self.config.embedding_model,
                    negative=self.config.negative_samples, epochs=self.config.epochs,
                    workers=self.config.cpu_count, learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                )
            return

        if backend == "pytorch":
            from .embedders.word2vec_loader import (
                InMemoryWalkSeqDataModule, WalkSequenceDataModule, walks_per_pair_batch,
            )
            walks_per_batch = walks_per_pair_batch(
                self.config.pair_batch_size, self.config.walk_depth, self.config.window_size
            )
            if self.config.multi_gpu:
                import glob as _glob
                import pyarrow.parquet as _pq
                n_gpus = torch.cuda.device_count()
                token_counts = _token_counts_from_walk_parquet(
                    walks_path, self.word2idx.shape[0]
                )
                # Real per-rank step count for SkipGram's LR schedule: total walks
                # (from parquet metadata, cheap) / (walks_per_batch * world_size).
                n_walks = sum(
                    _pq.read_metadata(f).num_rows
                    for f in _glob.glob(f"{walks_path}/*.parquet")
                ) or 1
                steps_per_epoch = max(1, -(-n_walks // (walks_per_batch * n_gpus)))
                dm = WalkSequenceDataModule(
                    walks_path, batch_size=walks_per_batch,
                    num_workers=self.config.cpu_count, shuffle=True, seed=self.config.random_state,
                )
            else:
                df = cudf.read_parquet(walks_path)
                n_gpus = 1
                token_counts = _token_counts_from_cudf_walks(df, self.word2idx.shape[0])
                steps_per_epoch = max(1, -(-int(len(df)) // walks_per_batch))
                dm = InMemoryWalkSeqDataModule(
                    df, emit_batch_size=walks_per_batch,
                    shuffle=True, seed=self.config.random_state,
                )
            with self.tracker.stage("Word2Vec_Training"):
                self._train_pytorch_sequences(
                    dm, token_counts=token_counts,
                    steps_per_epoch=steps_per_epoch, n_gpus=n_gpus,
                )
            return

        raise ValueError(f"fit_from_walks: unsupported backend {backend!r}")
```

> NOTE: `cudf.read_parquet(walks_path)` on a directory reads all part files. For the multi-GPU `steps_per_epoch`, Lightning re-derives the real step count from the streamed loader at runtime; the value passed only sizes SkipGram's LR schedule (see `_fit_via_pytorch_sequences` for the same pattern).

- [ ] **Step 4: Run the tests**

Run: `pytest test/embedders/fit_from_walks_test.py -k "missing_corpus or dispatches" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rdf2vecgpu/gpu_rdf2vec.py test/embedders/fit_from_walks_test.py
git commit -m "feat: fit_from_walks dispatches across gensim / pytorch (1+N GPU) / torchrec"
```

---

## Task 7: Integration — all four backends train from one shared corpus

**Files:**
- Test: `test/embedders/fit_from_walks_test.py`

- [ ] **Step 1: Write the single-process integration test** (gensim + pytorch single-GPU)

```python
@pytest.mark.gpu
def test_fit_from_walks_gensim_and_pytorch_single_from_shared_corpus(tmp_path):
    import cudf, numpy as np
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    triples = str(tmp_path / "t.parquet")
    cudf.DataFrame({
        "subject": [f"e{i%50}" for i in range(3000)],
        "predicate": [f"p{i%5}" for i in range(3000)],
        "object": [f"e{(i*3)%50}" for i in range(3000)],
    }).to_parquet(triples)

    common = dict(embedding_model="skipgram", walk_strategy="random",
                  walk_depth=3, walk_number=10, vector_size=16, epochs=1, multi_gpu=False)

    prep = GPU_RDF2Vec(config=RDF2VecConfig(backend="pytorch", **common))
    walks_path, vocab_path = prep.prepare_corpus(prep.load_data(triples), str(tmp_path / "corpus"))

    for backend in ("gensim", "pytorch"):
        m = GPU_RDF2Vec(config=RDF2VecConfig(backend=backend, **common))
        m.fit_from_walks(walks_path, vocab_path)
        emb = m.transform().to_pandas()
        vec = [c for c in emb.columns if c not in ("token", "word")]
        assert len(emb) > 0 and np.isfinite(emb[vec].to_numpy()).all()
```

- [ ] **Step 2: Run it**

Run: `pytest --gpu test/embedders/fit_from_walks_test.py::test_fit_from_walks_gensim_and_pytorch_single_from_shared_corpus -v`
Expected: PASS. (If gensim isn't installed: `pip install "rdf2vecgpu[gensim]"`.)

- [ ] **Step 3a: torchrec from-corpus integration test** (marker `multigpu`, under `torchrun`)

Add to `test/embedders/torchrec_word2vec_test.py` so it reuses `torchrun_pg` / `_prepare_corpus_fs` / `_gather_full_embeddings_fs`. (torchrec is already covered by `test_torchrec_recovers_planted_community_structure`; this just confirms the generalised `fit_from_walks` still routes torchrec correctly — a thin smoke reusing `_shared_base("ffw_torchrec")`.) If the existing torchrec test already exercises `fit_from_walks`, SKIP adding a duplicate and note that in the commit.

- [ ] **Step 3b: pytorch-mg from-corpus test** (plain `python`, Lightning spawns DDP — NOT under `torchrun`)

pytorch-mg's launch model is Lightning's internal DDP spawn from a single `python` process — it CANNOT run under the `torchrun_pg` fixture (that pre-inits a competing NCCL PG). Mirror the existing `test/e2e_multi_gpu_test.py` pattern instead: a plain test (own marker, e.g. `multigpu_ddp` registered in `pyproject.toml`, or run explicitly) that prepares a corpus single-process, then calls `fit_from_walks` with `multi_gpu=True` and lets Lightning spawn the DDP workers. Put it in `test/embedders/fit_from_walks_test.py`:

```python
@pytest.mark.gpu
def test_fit_from_walks_pytorch_mg_lightning_spawn(tmp_path):
    """pytorch-mg fit_from_walks: Lightning spawns DDP internally (plain python,
    no torchrun). Needs >=2 GPUs and NCCL_P2P_DISABLE=1 on PCIe boxes."""
    import os, cudf, numpy as np, torch
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    from rdf2vecgpu.config import RDF2VecConfig
    from rdf2vecgpu.gpu_rdf2vec import GPU_RDF2Vec

    triples = str(tmp_path / "t.parquet")
    cudf.DataFrame({
        "subject": [f"e{i%80}" for i in range(4000)],
        "predicate": [f"p{i%5}" for i in range(4000)],
        "object": [f"e{(i*7)%80}" for i in range(4000)],
    }).to_parquet(triples)
    common = dict(embedding_model="skipgram", walk_strategy="random",
                  walk_depth=3, walk_number=10, vector_size=16, epochs=1)

    prep = GPU_RDF2Vec(config=RDF2VecConfig(backend="pytorch", multi_gpu=False, **common))
    walks_path, vocab_path = prep.prepare_corpus(prep.load_data(triples), str(tmp_path / "corpus"))

    m = GPU_RDF2Vec(config=RDF2VecConfig(backend="pytorch", multi_gpu=True, **common))
    m.fit_from_walks(walks_path, vocab_path)     # Lightning Trainer(devices=N, ddp) spawns
    emb = m.transform().to_pandas()
    vec = [c for c in emb.columns if c not in ("token", "word")]
    assert np.isfinite(emb[vec].to_numpy()).all()
```

> NOTE: Lightning `strategy="ddp"` under pytest re-launches workers — confirm `test/e2e_multi_gpu_test.py` runs cleanly in this env first (it uses the same mechanism for the legacy multi-GPU path). If Lightning's subprocess launcher re-runs the test session, switch this test to a subprocess driver (`subprocess.run([python, "-m", driver_script, ...])`) modelled on `performance/bench_backends.py --backend pytorch-mg`, which is the canonical plain-python launch.

- [ ] **Step 4: Run the multi-GPU tests**

Run (torchrec): `NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m pytest -m multigpu test/embedders/torchrec_word2vec_test.py -p no:cacheprovider -o addopts="" -o faulthandler_timeout=200 -v`
Expected: all multigpu tests PASS.
Run (pytorch-mg): `NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 pytest --gpu test/embedders/fit_from_walks_test.py::test_fit_from_walks_pytorch_mg_lightning_spawn -v`
Expected: PASS (finite embeddings; no NCCL hang). If Lightning re-runs the session, use the subprocess-driver fallback in the note above (validated equivalently by the Task 8 bench smoke).

- [ ] **Step 5: Commit**

```bash
git add test/embedders/fit_from_walks_test.py test/embedders/torchrec_word2vec_test.py
git commit -m "test: fit_from_walks integration across all four backends from one corpus"
```

---

## Task 8: Benchmark harness — compare training-only from a shared corpus

**Files:**
- Modify: `performance/bench_backends.py`

- [ ] **Step 1: Rework `run_single_process` and `run_torchrec`** so each backend trains from a pre-prepared corpus instead of the full pipeline.

Add a `--corpus` argument (a directory produced by `prepare_corpus`). When set, all backends call `fit_from_walks(corpus/walks, corpus/vocab.parquet)` and time ONLY that call (+ `transform`). Add a `prepare` sub-mode that runs `prepare_corpus` once and writes the corpus, timed separately. For `pytorch-mg`, no Dask client is needed for `fit_from_walks` (training is pure DDP), so drop the `LocalCUDACluster`/`Comms` setup in that path. Concretely:

```python
def run_from_corpus(args):
    """All backends except torchrec: fit_from_walks from a shared corpus."""
    from rdf2vecgpu import GPU_RDF2Vec
    backend = "pytorch" if args.backend in ("pytorch", "pytorch-mg") else args.backend
    multi_gpu = args.backend == "pytorch-mg"
    cfg = _config(backend, multi_gpu, args)
    m = GPU_RDF2Vec(config=cfg)
    walks = os.path.join(args.corpus, "walks")
    vocab = os.path.join(args.corpus, "vocab.parquet")
    metrics = _common_metrics(args, args.backend)
    _reset_vram()
    with _timed(metrics, "t_fit_from_walks_s"):
        m.fit_from_walks(walks, vocab)
    metrics["peak_vram_gib"] = _peak_vram_gib()
    return metrics
```

For torchrec, `run_torchrec` already does `fit_from_walks`; point it at `args.corpus` instead of preparing its own. Add a top-level `--prepare` flag that runs `prepare_corpus` and exits (timed into a `prepare` row).

- [ ] **Step 2: Update `performance/README.md`** run commands:

```bash
# 1. prepare the corpus ONCE (shared by all backends)
python performance/bench_backends.py --prepare --data $DATA --corpus corpus/ --tag wikidata5m

# 2. compare TRAINING ONLY from that corpus
python performance/bench_backends.py --backend gensim     --corpus corpus/ --tag wikidata5m
python performance/bench_backends.py --backend pytorch    --corpus corpus/ --tag wikidata5m
NCCL_P2P_DISABLE=1 python performance/bench_backends.py --backend pytorch-mg --corpus corpus/ --tag wikidata5m
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 \
  performance/bench_backends.py --backend torchrec --corpus corpus/ --tag wikidata5m
```

- [ ] **Step 3: Smoke-test the harness** on the tiny graph (validate, don't run wikidata-5m)

```bash
python -c "import cudf,numpy as np; rng=np.random.default_rng(0); n=2000; \
cudf.DataFrame({'subject':[f'e{i}' for i in rng.integers(0,200,n)],'predicate':[f'p{i}' for i in rng.integers(0,10,n)],'object':[f'e{i}' for i in rng.integers(0,200,n)]}).to_parquet('/tmp/bt.parquet')"
python performance/bench_backends.py --prepare --data /tmp/bt.parquet --corpus /tmp/btc --walk-number 5 --walk-depth 3 --tag smoke
python performance/bench_backends.py --backend pytorch --corpus /tmp/btc --walk-number 5 --walk-depth 3 --epochs 1 --vector-size 16 --tag smoke
```
Expected: a `t_fit_from_walks_s` row in `performance/results/bench_results.jsonl`.

- [ ] **Step 4: Commit**

```bash
git add performance/bench_backends.py performance/README.md
git commit -m "perf(bench): compare training-only from a shared prepared corpus"
```

---

## Task 9: Docs

**Files:**
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 1:** In README's "Sharded embeddings across GPUs" section, add that `prepare_corpus` → `fit_from_walks(walks_path, vocab_path)` now works for **all** backends (not just torchrec): generate the corpus once, train any backend from it. Note that single-GPU pytorch `fit()` and `fit_from_walks` keep walks GPU-resident (DLPack), and that pytorch-mg/torchrec need `NCCL_P2P_DISABLE=1` on PCIe boxes.

- [ ] **Step 2:** Add a CHANGELOG entry under a new `## [Unreleased]` (or `0.6.0`) heading: uniform `fit_from_walks` across backends; restored single-GPU GPU-resident in-memory feed; fixed the gensim single-GPU `.persist()` crash and the bench harness's missing cuGraph-comms init.

- [ ] **Step 3: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: fit_from_walks is the cross-backend decoupled training entry"
```

---

## Self-review checklist (done before handing off)

- **Spec coverage:** Part A (uniform fit_from_walks) → Tasks 3, 6, 7. Part B (GPU-resident single-GPU) → Tasks 1, 2, 4, 5. Vocab loading → Task 3. Benchmark → Task 8. Docs → Task 9. ✓
- **Pre-existing bugs already fixed on-branch** (gensim `.persist()` discriminator; bench cuGraph-comms init) are committed; CHANGELOG note in Task 9 records them.
- **Type/name consistency:** `InMemoryWalkSeqDataModule(cudf_df, *, emit_batch_size, shuffle, seed)`, `_token_counts_from_cudf_walks(cudf_df, vocab_size)`, `_load_word2idx(vocab_path)`, `_train_pytorch_sequences(datamodule, *, token_counts, steps_per_epoch, n_gpus)` are used identically across Tasks 1–8.
- **Open implementer note:** Tasks 4/5 say to copy the LIVE `SkipGram(...)` / `L.Trainer(...)` kwargs from `_fit_via_pytorch_sequences` rather than the illustrative snippets, so behaviour stays byte-identical.
