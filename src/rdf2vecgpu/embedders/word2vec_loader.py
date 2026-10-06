import glob
import queue
import threading
import lightning as L
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    BatchSampler,
    IterableDataset,
    get_worker_info,
)
import torch


class _IndexDataset(Dataset):
    """Lightweight wrapper that just returns its own index.

    The tensors live outside the dataset; DataLoader gives us a list of
    integer indices that we slice in a custom collate_fn.  Returning only the
    index keeps __getitem__ O(1) and avoids creating per‑sample tensors.
    """

    def __init__(self, length: int):
        self._len = length

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):  # noqa: D401, D403  (simple stub)
        return idx  # int – cheap to move between Python and C++


class SkipGramDataModule(L.LightningDataModule):
    """Dataloading optimised for a GPU‑resident skip‑gram table.

    Parameters
    ----------
    center_tensor, context_tensor
        1‑D CUDA tensors with the same length.
    batch_size
        Number of (centre, context) pairs per optimisation step.
    """

    def __init__(
        self,
        center_tensor: torch.Tensor,
        context_tensor: torch.Tensor,
        *,
        batch_size: int,
    ):
        super().__init__()
        assert center_tensor.device.type == "cuda", "tensors must be on GPU"
        assert center_tensor.shape == context_tensor.shape

        self.center = center_tensor.contiguous()
        self.context = context_tensor.contiguous()
        self.batch_size = batch_size

        # length is reused in sampler
        self._dataset = _IndexDataset(len(self.center))

    def setup(self, stage: str | None = None):
        pass

    def train_dataloader(self):
        sampler = RandomSampler(self._dataset, replacement=False)
        batch_sampler = BatchSampler(
            sampler, batch_size=self.batch_size, drop_last=False
        )

        def _collate(indices: list[int]):  # indices comes from BatchSampler
            idx = torch.tensor(indices, device=self.center.device)
            return self.center[idx], self.context[idx]

        return DataLoader(
            self._dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate,
            num_workers=0,  # CUDA tensors + index dataset → no workers
            pin_memory=False,
        )


class CBOWDataModule(L.LightningDataModule):
    """Dataloading optimised for a GPU-resident CBOW table.

    Parameters
    ----------
    context_tensor : torch.Tensor
        2-D CUDA tensor of shape (n_samples, ctx_size), where each row
        is the flattened context words for one target.
    center_tensor : torch.Tensor
        1-D CUDA tensor of length n_samples, the target (centre) word indices.
    batch_size : int
        Number of (context_vec, center) samples per optimisation step.
    """

    def __init__(
        self,
        context_tensor: torch.Tensor,
        center_tensor: torch.Tensor,
        *,
        batch_size: int,
    ):
        super().__init__()
        # both tensors must be on CUDA
        assert context_tensor.device.type == "cuda", "context_tensor must be on GPU"
        assert center_tensor.device.type == "cuda", "center_tensor must be on GPU"
        # must have same first dimension
        assert context_tensor.dim() == 2, "context_tensor must be 2-D"
        assert center_tensor.dim() == 1, "center_tensor must be 1-D"
        assert context_tensor.shape[0] == center_tensor.shape[0]

        # store contiguous views
        self.context = context_tensor.contiguous()
        self.center = center_tensor.contiguous()
        self.batch_size = batch_size

        # simple index dataset
        self._dataset = _IndexDataset(len(self.center))

    def setup(self, stage: str | None = None):
        # nothing extra to do here
        pass

    def train_dataloader(self):
        sampler = RandomSampler(self._dataset, replacement=False)
        batch_sampler = BatchSampler(
            sampler, batch_size=self.batch_size, drop_last=False
        )

        def _collate(indices: list[int]):
            idx = torch.tensor(indices, device=self.center.device, dtype=torch.long)
            # contexts: [B, ctx_size], centers: [B]
            return self.context[idx], self.center[idx]

        return DataLoader(
            self._dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate,
            num_workers=0,  # GPU tensors + index sampler → no subprocesses
            pin_memory=False,
        )


class ParquetSkipGramDataModule(L.LightningDataModule):
    """DataModule that reads skip-gram pairs from partitioned parquet.

    Designed for multi-GPU DDP training: each rank reads its own subset
    of parquet files and loads them onto its local GPU.

    Parameters
    ----------
    parquet_path : str
        Directory containing partitioned parquet files (written by dask).
    batch_size : int
        Number of (centre, context) pairs per optimisation step.
    """

    def __init__(self, parquet_path: str, *, batch_size: int):
        super().__init__()
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.center = None
        self.context = None

    def setup(self, stage: str | None = None):
        import cudf
        from glob import glob
        import os

        files = sorted(glob(os.path.join(self.parquet_path, "*.parquet")))
        if not files:
            # dask may write to subdirectory "part.0.parquet" etc.
            files = sorted(glob(os.path.join(self.parquet_path, "**", "*.parquet"), recursive=True))

        # Distribute files across DDP ranks
        rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1
        my_files = [f for i, f in enumerate(files) if i % world_size == rank]

        if not my_files:
            raise RuntimeError(
                f"Rank {rank}/{world_size} has no parquet files assigned "
                f"from {len(files)} total files in {self.parquet_path}"
            )

        # Read assigned parquet files and convert to GPU tensors
        parts = [cudf.read_parquet(f) for f in my_files]
        df = cudf.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        self.center = torch.utils.dlpack.from_dlpack(
            df["center"].to_dlpack()
        ).contiguous()
        self.context = torch.utils.dlpack.from_dlpack(
            df["context"].to_dlpack()
        ).contiguous()

    def train_dataloader(self):
        dataset = _IndexDataset(len(self.center))
        sampler = RandomSampler(dataset, replacement=False)
        batch_sampler = BatchSampler(
            sampler, batch_size=self.batch_size, drop_last=False
        )

        def _collate(indices: list[int]):
            idx = torch.tensor(indices, device=self.center.device)
            return self.center[idx], self.context[idx]

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate,
            num_workers=0,
            pin_memory=False,
        )


class OrderAwareSkipGramDataModule(L.LightningDataModule):
    """
    Dataloading optimised for a GPU‑resident order-aware skip‑gram table.
    Collates batches of (center, context, distance).
    """

    def __init__(
        self,
        center_tensor: torch.Tensor,
        context_tensor: torch.Tensor,
        distance_tensor: torch.Tensor,
        *,
        batch_size: int,
    ):
        super().__init__()
        # Ensure everything is on GPU to avoid transfers during training
        assert center_tensor.device.type == "cuda", "tensors must be on GPU"
        assert context_tensor.device.type == "cuda", "tensors must be on GPU"
        assert distance_tensor.device.type == "cuda", "tensors must be on GPU"

        # Validation: All arrays must have the same number of training pairs
        assert center_tensor.shape == context_tensor.shape == distance_tensor.shape

        self.center = center_tensor.contiguous()
        self.context = context_tensor.contiguous()
        self.distance = distance_tensor.contiguous()
        self.batch_size = batch_size

        self._dataset = _IndexDataset(len(self.center))

    def train_dataloader(self):
        sampler = RandomSampler(self._dataset, replacement=False)
        batch_sampler = BatchSampler(
            sampler, batch_size=self.batch_size, drop_last=False
        )

        def _collate(indices: list[int]):
            # 1. Move indices to GPU
            idx = torch.tensor(indices, device=self.center.device)
            # 2. Slice the large GPU tensors directly
            # Returns: (Batch, ), (Batch, ), (Batch, )
            return self.center[idx], self.context[idx], self.distance[idx]

        return DataLoader(
            self._dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate,
            num_workers=0,
            pin_memory=False,
        )


class OrderAwareCBOWDataModule(L.LightningDataModule):
    """
    Dataloading optimised for a GPU-resident order-aware CBOW table.
    Collates batches of (context_words, context_distances, center_word).
    """

    def __init__(
        self,
        context_tensor: torch.Tensor,  # Shape: (N, WindowSize)
        context_distance_tensor: torch.Tensor,  # Shape: (N, WindowSize)
        center_tensor: torch.Tensor,  # Shape: (N,)
        *,
        batch_size: int,
    ):
        super().__init__()
        assert context_tensor.device.type == "cuda"
        assert context_distance_tensor.device.type == "cuda"
        assert center_tensor.device.type == "cuda"

        assert context_tensor.dim() == 2, "Context must be 2D matrix"
        assert context_distance_tensor.dim() == 2, "Distance must be 2D matrix"

        # Validation
        assert context_tensor.shape == context_distance_tensor.shape
        assert context_tensor.shape[0] == center_tensor.shape[0]

        self.context = context_tensor.contiguous()
        self.context_distance = context_distance_tensor.contiguous()
        self.center = center_tensor.contiguous()
        self.batch_size = batch_size

        self._dataset = _IndexDataset(len(self.center))

    def train_dataloader(self):
        sampler = RandomSampler(self._dataset, replacement=False)
        batch_sampler = BatchSampler(
            sampler, batch_size=self.batch_size, drop_last=False
        )

        def _collate(indices: list[int]):
            idx = torch.tensor(indices, device=self.center.device, dtype=torch.long)
            # Returns: (Batch, Win), (Batch, Win), (Batch,)
            return self.context[idx], self.context_distance[idx], self.center[idx]

        return DataLoader(
            self._dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate,
            num_workers=0,
            pin_memory=False,
        )


# --------------------------------------------------------------------------- #
# Walk-sequence streaming (PyTorch backend's on-GPU pair-expansion corpus path)
# --------------------------------------------------------------------------- #


def skipgram_pairs_per_walk(walk_depth: int, window: int) -> int:
    """Ordered (center, context) skip-gram pairs a *full-length* walk of depth
    ``walk_depth`` produces with context window ``window``. The walk is the
    interleaved vertices+predicates sequence of length ``L = 2 * walk_depth + 1``;
    a pair ``(i, j)`` is produced for every ``0 <= i, j < L`` with
    ``0 < |i - j| <= window``. (cuGraph-truncated short walks produce fewer
    pairs — this is the upper bound.)"""
    L = 2 * walk_depth + 1
    total = 0
    for i in range(L):
        lo = max(0, i - window)
        hi = min(L - 1, i + window)
        total += (hi - lo + 1) - 1  # in-window positions, minus i itself
    return total


def walks_per_pair_batch(pair_batch_size: int, walk_depth: int, window: int) -> int:
    """How many walk rows to put in one yielded training batch so that, after
    on-GPU pair expansion, the batch holds about ``pair_batch_size`` skip-gram
    pairs. Floors the division (so the realised batch is <= target) and never
    returns 0."""
    return max(1, pair_batch_size // skipgram_pairs_per_walk(walk_depth, window))


def _flat_list_array_to_2d(col):
    """A pyarrow list<int> column where every row has the same length →
    a contiguous (n_rows, row_len) int32 numpy array — no per-row Python
    (uses the flat .values + .offsets, not to_pylist())."""
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    n = len(arr)
    if n == 0:
        return np.empty((0, 0), dtype=np.int32)
    offsets = arr.offsets.to_numpy(zero_copy_only=False)
    row_len = int(offsets[1] - offsets[0])
    if not np.all(np.diff(offsets) == row_len):
        return np.asarray(arr.to_pylist(), dtype=np.int32)  # ragged — defensive fallback
    flat = arr.values.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    return flat.reshape(n, row_len)


class _WalkSeqIterable(IterableDataset):
    """Streams a walks-as-sequences parquet directory. Reads each file in large
    `read_chunk_size`-row PyArrow batches (cheap), decodes them vectorised, then
    re-slices into `emit_batch_size`-row sub-batches and yields each as
    (vertices, predicates) numpy int32 arrays of shape (n, depth+1) / (n, depth).
    Single-rank (world_size=1): files split across DataLoader workers, all rows
    emitted. DDP (world_size>1): each rank row-strides a disjoint 1/world_size of
    the corpus, capped so every rank emits the same number of batches (required
    because DDP all_reduces gradients every step)."""

    def __init__(self, walks_dir, *, emit_batch_size, read_chunk_size=262_144,
                 shuffle=True, seed=0, rank=0, world_size=1,
                 pair_batch_size=None, window=5, drop_stubs=None):
        super().__init__()
        # Optional PAIR-BUDGET mode (torchrec path): cut batches by cumulative
        # skip-gram pair count instead of a fixed row count, so every batch
        # carries ~pair_batch_size pairs however short the walks are. Measured
        # 2026-09-05: Wikidata walks yield 17 of the nominal 60 pairs (40% are
        # single-vertex stubs), dbpedia2016 19 — so a fixed row count delivered
        # under a third of the configured batch and 3.5× the steps. Cut points
        # depend only on the (rank-independent) row stream, so ranks that stride
        # rows afterwards (see _PairsFromWalks) keep identical batch counts.
        # Not combinable with the in-iterable DDP striding (world_size > 1):
        # there the rows are rank-strided BEFORE batching and the cut points
        # would differ per rank.
        self.pair_batch_size = int(pair_batch_size) if pair_batch_size else None
        self.window = int(window)
        if self.pair_batch_size and int(world_size) > 1:
            raise ValueError(
                "pair_batch_size batching needs world_size=1 in the iterable; "
                "stride rows per rank afterwards (see _PairsFromWalks)"
            )
        # drop_stubs: discard walks with fewer than two valid vertices right
        # after decode. They yield no skip-gram pairs, so the pairs and the batch
        # count are unchanged (with shuffle on, the within-chunk permutation then
        # runs over the surviving rows — a different, equally valid shuffle); but stubs
        # are ~40% of Wikidata's rows and would otherwise ride through the
        # shuffle, the batch concatenation and the pair expansion. Applied on
        # the same rank-independent stream as the pair cuts, hence the same
        # world_size=1 restriction; defaults to on in pair-budget mode and off
        # otherwise (row-count consumers expect every row).
        self.drop_stubs = bool(self.pair_batch_size) if drop_stubs is None else bool(drop_stubs)
        if self.drop_stubs and int(world_size) > 1:
            raise ValueError("drop_stubs needs world_size=1 in the iterable (see pair_batch_size)")
        files = sorted(set(
            glob.glob(f"{walks_dir}/*.parquet")
            + glob.glob(f"{walks_dir}/**/*.parquet", recursive=True)
        ))
        if not files:
            raise FileNotFoundError(f"no parquet under {walks_dir}")
        self.files = files
        self.emit_batch_size = int(emit_batch_size)
        self.read_chunk_size = int(read_chunk_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        # For DDP (world_size > 1) we row-stride the corpus across ranks and cap
        # every rank to the same row budget, so each rank emits an identical
        # number of batches (the per-step gradient all_reduce would otherwise
        # desync and hang). That needs the global row count up front — cheap, it
        # comes from parquet metadata without reading any data.
        self.total_rows = (
            sum(pq.ParquetFile(f).metadata.num_rows for f in self.files)
            if self.world_size > 1 else None
        )
        # epoch counter so successive epochs (Lightning re-iterates the dataloader)
        # get a different shuffle. Bumped in __iter__.
        self._epoch = -1

    def __iter__(self):
        wi = get_worker_info()
        n_w, w_id = (wi.num_workers, wi.id) if wi else (1, 0)
        self._epoch += 1
        if self.world_size > 1:
            yield from self._iter_ddp_strided(n_w, w_id)
        else:
            yield from self._iter_single(n_w, w_id)

    def _maybe_shuffle(self, v, p, *, shard_id, f_idx, c_idx):
        # cuGraph emits the ~walk_number walks of each start vertex consecutively;
        # without this shuffle a vertex's walks clump into one or two batches
        # (over-aggregating its gradient / hitting the grad clip, then absent the
        # rest of the epoch). Shuffling within the (large) read chunk spreads each
        # vertex's walks across ~read_chunk_size/emit_batch_size batches. Seed mixes
        # shard id, file index, chunk index and epoch so it's deterministic-per-run
        # but varies across.
        if not self.shuffle or v.shape[0] == 0:
            return v, p
        rng = np.random.default_rng((self.seed, self._epoch, shard_id, f_idx, c_idx))
        perm = rng.permutation(v.shape[0])
        return v[perm], p[perm]

    def _iter_single(self, n_w, w_id):
        """Single-rank streaming: files split across DataLoader workers, all rows."""
        eb = self.emit_batch_size
        batcher = (
            _PairBudgetBatcher(self.pair_batch_size, self.window)
            if self.pair_batch_size else None
        )
        my = [f for k, f in enumerate(self.files) if k % n_w == w_id]
        for f_idx, f in enumerate(my):
            pf = pq.ParquetFile(f)
            for c_idx, b in enumerate(
                pf.iter_batches(columns=["vertices", "predicates"], batch_size=self.read_chunk_size)
            ):
                v = _flat_list_array_to_2d(b.column("vertices"))
                p = _flat_list_array_to_2d(b.column("predicates"))
                if self.drop_stubs and v.shape[0]:
                    keep = _walk_valid_len(v) >= 3      # < 2 valid vertices → 0 pairs
                    if not keep.all():
                        v, p = v[keep], p[keep]
                if v.shape[0] == 0:
                    continue
                v, p = self._maybe_shuffle(v, p, shard_id=w_id, f_idx=f_idx, c_idx=c_idx)
                if batcher is not None:
                    yield from batcher.push(v, p)
                else:
                    for s in range(0, v.shape[0], eb):
                        yield v[s:s + eb], p[s:s + eb]
        if batcher is not None:
            yield from batcher.flush()

    def _iter_ddp_strided(self, n_w, w_id):
        """DDP streaming: each (rank, worker) owns a disjoint global row-stride and
        emits exactly ``total_rows // (world_size * n_w)`` rows, so every rank emits
        the same number of batches. Rows are buffered across read chunks and emitted
        in full ``emit_batch_size`` blocks (only the final block is ragged) so the
        batch COUNT depends only on the per-shard cap — never on chunk boundaries."""
        eb = self.emit_batch_size
        n_shards = self.world_size * n_w
        shard_id = self.rank * n_w + w_id
        cap = self.total_rows // n_shards          # rows this shard trains on
        if cap == 0:
            return
        emitted = 0
        g = 0                                      # global row offset across all files
        carry_v = carry_p = None
        for f_idx, f in enumerate(self.files):
            pf = pq.ParquetFile(f)
            for c_idx, b in enumerate(
                pf.iter_batches(columns=["vertices", "predicates"], batch_size=self.read_chunk_size)
            ):
                v = _flat_list_array_to_2d(b.column("vertices"))
                p = _flat_list_array_to_2d(b.column("predicates"))
                m = v.shape[0]
                if m == 0:
                    continue
                # rows in [g, g+m) owned by this shard: (g + local) % n_shards == shard_id
                start = (shard_id - g) % n_shards
                if start < m:
                    owned = np.arange(start, m, n_shards)
                    take = min(len(owned), cap - emitted)
                    owned = owned[:take]
                    if len(owned):
                        ov, op = v[owned], p[owned]
                        ov, op = self._maybe_shuffle(ov, op, shard_id=shard_id, f_idx=f_idx, c_idx=c_idx)
                        carry_v = ov if carry_v is None else np.concatenate([carry_v, ov])
                        carry_p = op if carry_p is None else np.concatenate([carry_p, op])
                        emitted += len(owned)
                        while carry_v.shape[0] >= eb:
                            yield carry_v[:eb], carry_p[:eb]
                            carry_v, carry_p = carry_v[eb:], carry_p[eb:]
                g += m
                if emitted >= cap:
                    break
            if emitted >= cap:
                break
        if carry_v is not None and carry_v.shape[0] > 0:
            yield carry_v, carry_p


def _walkseq_collate(item):
    v, p = item  # already a (vertices_np, predicates_np) tuple from _WalkSeqIterable
    return torch.from_numpy(v), torch.from_numpy(p)


# --------------------------------------------------------------------------- #
# Skip-gram pair accounting + torchrec pair loader stages
# --------------------------------------------------------------------------- #


def _pairs_by_valid_len(seq_len: int, window: int) -> np.ndarray:
    """``table[L]`` = skip-gram pairs a walk yields when the first ``L`` of its
    ``seq_len`` interleaved (vertex, predicate, …) tokens are valid and the
    rest are ``-1``: pairs ``(i, j)`` with ``0 < |i - j| <= window`` and both
    ``< L``. Mirrors :class:`_PairsFromWalks`' template exactly, and
    ``table[seq_len] == skipgram_pairs_per_walk(depth, window)``."""
    table = np.zeros(seq_len + 1, dtype=np.int64)
    for L in range(seq_len + 1):
        table[L] = sum(min(L, i + window + 1) - max(0, i - window) - 1 for i in range(L))
    return table


def _walk_valid_len(v: np.ndarray) -> np.ndarray:
    """Per-row valid length of the interleaved sequence: index of the first
    ``-1`` vertex, doubled, minus the trailing predicate (``v``: ``(n,
    depth+1)`` with cuGraph's ``-1`` sentinels for short walks)."""
    depth_p1 = v.shape[1]
    seq_len = 2 * depth_p1 - 1
    is_sent = v == -1
    first_sent = np.where(is_sent.any(axis=1), is_sent.argmax(axis=1), depth_p1)
    return np.clip(2 * first_sent - 1, 0, seq_len)


def _walk_pair_counts(v: np.ndarray, window: int) -> np.ndarray:
    """Skip-gram pairs each walk row of ``v`` yields under ``window``."""
    return _pairs_by_valid_len(2 * v.shape[1] - 1, window)[_walk_valid_len(v)]


class _PairBudgetBatcher:
    """Re-cuts a stream of walk rows into batches holding ~``pair_batch_size``
    skip-gram pairs. The walk that crosses the budget is included, so a batch
    overshoots by less than one walk's pairs; rows are never reordered or
    dropped. Cut points depend only on the row stream — never on a rank — so
    DDP ranks that later take disjoint row strides of every batch still see
    identical batch counts."""

    def __init__(self, pair_batch_size: int, window: int):
        self.budget = int(pair_batch_size)
        self.window = int(window)
        self._v, self._p, self._pairs = [], [], 0

    def push(self, v, p):
        counts = _walk_pair_counts(v, self.window)
        cum = np.cumsum(counts)
        start = 0
        while True:
            base = int(cum[start - 1]) if start > 0 else 0
            need = self.budget - self._pairs
            k = int(np.searchsorted(cum, base + need, side="left"))  # cum[k] >= base+need
            if k >= len(cum):
                break
            self._v.append(v[start:k + 1])
            self._p.append(p[start:k + 1])
            yield self._pop()
            start = k + 1
        if start < len(counts):
            self._v.append(v[start:])
            self._p.append(p[start:])
            self._pairs += int(cum[-1]) - (int(cum[start - 1]) if start > 0 else 0)

    def _pop(self):
        v = self._v[0] if len(self._v) == 1 else np.concatenate(self._v)
        p = self._p[0] if len(self._p) == 1 else np.concatenate(self._p)
        self._v, self._p, self._pairs = [], [], 0
        return v, p

    def flush(self):
        if self._v:
            yield self._pop()


class _PairsFromWalks(IterableDataset):
    """Expand each (vertices, predicates) walk batch into a flat (center,
    context) int64 pair batch on the CPU.

    In DDP every rank consumes the SAME inner stream and keeps a disjoint row
    stride of every batch, so all ranks yield the same number of batches —
    torchrec's sharded forward issues an all_to_all every step and deadlocks
    otherwise. An empty stride (tiny tail batch) is still yielded: the SUM
    loss contributes 0 but the rank joins the collective.
    """

    def __init__(self, inner, *, window, rank=0, world_size=1):
        super().__init__()
        self.inner = inner
        self.window = int(window)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._templates = {}

    def _template(self, seq_len):
        t = self._templates.get(seq_len)
        if t is None:
            w = self.window
            ii, jj = [], []
            for i in range(seq_len):
                for j in range(max(0, i - w), min(seq_len, i + w + 1)):
                    if j != i:
                        ii.append(i)
                        jj.append(j)
            t = (np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64))
            self._templates[seq_len] = t
        return t

    @staticmethod
    def expand(v_np, p_np, ii, jj):
        n, depth_p1 = v_np.shape
        seq_len = 2 * depth_p1 - 1
        seq = np.full((n, seq_len), -1, dtype=np.int64)
        seq[:, 0::2] = v_np
        seq[:, 1::2] = p_np
        valid_len = _walk_valid_len(v_np)
        keep = (
            (ii[None, :] < valid_len[:, None]) & (jj[None, :] < valid_len[:, None])
        ).reshape(-1)
        return seq[:, ii].reshape(-1)[keep], seq[:, jj].reshape(-1)[keep]

    def __iter__(self):
        for v_np, p_np in self.inner:
            if self.world_size > 1:
                v_np = v_np[self.rank::self.world_size]
                p_np = p_np[self.rank::self.world_size]
            ii, jj = self._template(2 * v_np.shape[1] - 1)
            c, x = self.expand(v_np, p_np, ii, jj)
            yield torch.from_numpy(c), torch.from_numpy(x)


class _PrefetchError:
    def __init__(self, exc):
        self.exc = exc


class _ThreadPrefetch(IterableDataset):
    """Run ``inner``'s iterator in a daemon thread, ``depth`` items ahead, so
    the parquet decode + pair expansion overlap the GPU step. Same items, same
    order; an exception in the producer is re-raised in the consumer; a
    consumer that stops early releases the producer. A thread rather than a
    DataLoader worker process: the consumer has CUDA + NCCL initialised
    (fork-unsafe), and the heavy parts (pyarrow decode, large numpy copies)
    release the GIL."""

    def __init__(self, inner, *, depth=4):
        super().__init__()
        self.inner = inner
        self.depth = int(depth)

    def __iter__(self):
        q = queue.Queue(maxsize=self.depth)
        stop = threading.Event()
        done = object()

        def put(item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def produce():
            try:
                for item in self.inner:
                    if not put(item):
                        return
                put(done)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
                put(_PrefetchError(exc))

        t = threading.Thread(target=produce, name="rdf2vec-prefetch", daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is done:
                    return
                if isinstance(item, _PrefetchError):
                    raise item.exc
                yield item
        finally:
            stop.set()


class WalkSequenceDataModule(L.LightningDataModule):
    """Feeds (vertices, predicates) int32 batches from a walks-as-sequences parquet
    dir. `num_workers` overlaps parquet read+decode with the GPU step; in DDP the
    file list is sharded per rank in setup()."""

    def __init__(self, walks_dir, *, batch_size, num_workers=0, read_chunk_size=262_144,
                 shuffle=True, seed=0):
        super().__init__()
        self.walks_dir = walks_dir
        self.batch_size = batch_size            # walks per yielded training batch
        self.num_workers = num_workers
        self.read_chunk_size = read_chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self._iterable = None

    def setup(self, stage=None):
        # The iterable row-strides across DDP ranks itself (each rank trains on a
        # disjoint 1/world_size of the rows, with equal batch counts) — so a
        # single-file corpus from single-GPU prepare_corpus shards correctly
        # without any redundant all-ranks-all-files replication.
        rank = self.trainer.global_rank if self.trainer is not None else 0
        world_size = self.trainer.world_size if self.trainer is not None else 1
        self._iterable = _WalkSeqIterable(
            self.walks_dir,
            emit_batch_size=self.batch_size,
            read_chunk_size=self.read_chunk_size,
            shuffle=self.shuffle,
            seed=self.seed,
            rank=rank,
            world_size=world_size,
        )

    def train_dataloader(self):
        if self._iterable is None:
            self.setup()
        return DataLoader(
            self._iterable,
            batch_size=None,                 # the iterable yields already-batched items
            collate_fn=_walkseq_collate,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )


# --------------------------------------------------------------------------- #
# GPU-resident in-memory walk-sequence DataModule (single-GPU fast path)
# --------------------------------------------------------------------------- #


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
        ).contiguous().view(n, -1) if n else torch.empty((0, 0), dtype=torch.int32, device="cuda")
        self.predicates = torch.from_dlpack(
            cudf_df["predicates"].list.leaves.to_dlpack()
        ).contiguous().view(n, -1) if n else torch.empty((0, 0), dtype=torch.int32, device="cuda")
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


# --------------------------------------------------------------------------- #
# Walk-corpus token counts (unigram negative-sampling weights) + on-disk cache
# --------------------------------------------------------------------------- #


def _file_token_uniques(fp):
    """One whole walks shard -> ``(unique token ids, counts)``; see
    :func:`_chunk_token_uniques` for the bounded-memory form the corpus
    pass uses. Kept for callers that hand in small files."""
    md = pq.ParquetFile(fp).metadata
    return _chunk_token_uniques(fp, 0, md.num_row_groups)


def _chunk_token_uniques(fp, rg_lo, rg_hi):
    """Row groups ``[rg_lo, rg_hi)`` of one walks shard -> ``(unique token ids,
    counts)`` over the ``vertices`` + ``predicates`` list columns, ``-1``
    cuGraph sentinels dropped. Module-level (not a closure) so it pickles into
    worker processes. Reading row groups rather than whole files is what keeps
    memory bounded: the 2026-10-06 dedup corpus ships 1.3 GB shards (47.5M
    walks each), and a whole-file read plus flatten/unique peaked at several
    GB per worker — 48 workers were OOM-killed at 120G in one minute."""
    pf = pq.ParquetFile(fp)
    parts = []
    for rg in range(rg_lo, rg_hi):
        tbl = pf.read_row_group(rg, columns=["vertices", "predicates"])
        for col in ("vertices", "predicates"):
            flat = tbl.column(col).combine_chunks().flatten().to_numpy(zero_copy_only=False)
            # A list-flattened column can come back float64 (NaN for nulls) when the
            # chunked array carries any null handling; bincount-style code needs ints.
            if np.issubdtype(flat.dtype, np.floating):
                flat = flat[~np.isnan(flat)]
            flat = flat.astype(np.int64, copy=False)
            parts.append(flat[flat >= 0])
    flat = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    if flat.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.unique(flat, return_counts=True)


def _plan_token_count_chunks(files, rows_per_chunk):
    """``(file, rg_lo, rg_hi)`` work items covering every row group of every
    file, each holding about ``rows_per_chunk`` walks (a single oversized row
    group is its own item). Parquet metadata only — no data is read."""
    items = []
    for fp in files:
        md = pq.ParquetFile(fp).metadata
        n, lo = md.num_row_groups, 0
        while lo < n:
            hi, rows = lo, 0
            while hi < n and (rows == 0 or rows + md.row_group(hi).num_rows <= rows_per_chunk):
                rows += md.row_group(hi).num_rows
                hi += 1
            items.append((fp, lo, hi))
            lo = hi
    return items


def _count_chunks_into(items, vocab_size, out_path=None):
    """Accumulate ``items`` (see :func:`_plan_token_count_chunks`) into one
    vocab-sized int64 array; save it to ``out_path`` when given (worker
    processes hand their partial back through a file, not a 3 GB pickle).
    ``u`` is unique within a chunk, so a plain fancy-index add is exact."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    for fp, lo, hi in items:
        u, c = _chunk_token_uniques(fp, lo, hi)
        keep = u < vocab_size
        counts[u[keep]] += c[keep]
    if out_path is not None:
        np.save(out_path, counts)
        return None
    return counts


def token_counts_from_walk_parquet(walks_parquet, vocab_size, *, workers=1, rows_per_chunk=4_000_000):
    """Token occurrence counts over a walk-sequences parquet dir -> int64 array of
    length ``vocab_size`` (token id -> count). Feeds the f^0.75 unigram negative
    sampler and the walked-coverage mask.

    The corpus is cut into row-group chunks of about ``rows_per_chunk`` walks
    (:func:`_plan_token_count_chunks`); each chunk is read, flattened and
    ``np.unique``-counted, then added into a vocab-sized accumulator. With
    ``workers > 1`` the chunk list is dealt round-robin to that many spawned
    processes, each with its OWN accumulator written to a temp file at the end
    and summed here — so peak memory is ``workers x (8 B x vocab_size + a
    chunk's working set)``, independent of shard size: ~4.5 GB per worker at
    the 393M Wikidata vocab, versus the per-file form that read 1.3 GB shards
    whole and was OOM-killed. (The earlier per-file ``np.unique`` replaced a
    per-file ``bincount(minlength=vocab_size)`` that allocated 3 GB per call.)
    """
    files = sorted(glob.glob(f"{walks_parquet}/*.parquet")) or [walks_parquet]
    items = _plan_token_count_chunks(files, int(rows_per_chunk))
    workers = max(1, min(int(workers), len(items)))
    if workers <= 1:
        return _count_chunks_into(items, vocab_size)

    import multiprocessing
    import os
    import tempfile
    ctx = multiprocessing.get_context("spawn")       # safe after CUDA init, unlike fork
    with tempfile.TemporaryDirectory(prefix="token_counts_") as tmp:
        procs = []
        for w in range(workers):
            out = os.path.join(tmp, f"part_{w}.npy")
            p = ctx.Process(target=_count_chunks_into, args=(items[w::workers], vocab_size, out), name=f"token-counts-{w}")
            p.start()
            procs.append((p, out))
        counts = np.zeros(vocab_size, dtype=np.int64)
        failed = []
        for p, out in procs:
            p.join()
            if p.exitcode != 0 or not os.path.exists(out):
                failed.append((p.name, p.exitcode))
                continue
            counts += np.load(out)
        if failed:
            raise RuntimeError(f"token count workers failed (name, exitcode): {failed}")
    return counts


def token_counts_cache_path(vocab_path):
    """``token_counts.npy`` next to the corpus vocab — the cache key is the vocab
    file, since token ids only mean anything relative to it."""
    import os
    return os.path.join(os.path.dirname(os.path.abspath(vocab_path)), "token_counts.npy")


def load_token_counts_cache(path, vocab_size):
    """Cached counts, or ``None`` if absent / not int64 / wrong length (a stale
    cache from a different vocab must never be silently reused)."""
    import os
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.ndim != 1 or arr.shape[0] != vocab_size or arr.dtype != np.int64:
        return None
    return arr


def save_token_counts_cache(path, counts):
    """Atomic write (tmp + rename) so a reader never sees a half-written file."""
    import os
    tmp = f"{path}.{os.getpid()}.tmp.npy"      # .npy suffix so np.save doesn't append one
    np.save(tmp, np.ascontiguousarray(counts, dtype=np.int64))
    os.replace(tmp, path)
