import glob

import lightning as L
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class _GPUBatches(IterableDataset):
    """Batches of rows of GPU tensors, in a new random order each epoch.

    The order is one `torch.randperm` on the GPU per epoch and each batch is a
    slice of it, so no Python work grows with the number of rows. (A
    `RandomSampler` + `BatchSampler` builds a Python list of every row index
    per epoch, which made training wait on the CPU.)
    """

    def __init__(self, tensors, batch_size: int):
        super().__init__()
        self.tensors = tensors
        self.batch_size = int(batch_size)

    def __len__(self):
        return -(-len(self.tensors[0]) // self.batch_size)

    def __iter__(self):
        n = len(self.tensors[0])
        order = torch.randperm(n, device=self.tensors[0].device)
        for start in range(0, n, self.batch_size):
            rows = order[start : start + self.batch_size]
            yield tuple(t[rows] for t in self.tensors)


def _gpu_batch_loader(tensors, batch_size: int) -> DataLoader:
    # the dataset yields ready batches of CUDA tensors: no collation, and no
    # worker processes (CUDA tensors cannot cross them)
    return DataLoader(_GPUBatches(tensors, batch_size), batch_size=None, num_workers=0)


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

    def setup(self, stage: str | None = None):
        pass

    def train_dataloader(self):
        return _gpu_batch_loader((self.center, self.context), self.batch_size)


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

    def setup(self, stage: str | None = None):
        pass

    def train_dataloader(self):
        # contexts: [B, ctx_size], centers: [B]
        return _gpu_batch_loader((self.context, self.center), self.batch_size)


class ParquetSkipGramDataModule(L.LightningDataModule):
    """DataModule that reads skip-gram pairs from partitioned parquet.

    Designed for multi-GPU DDP training: each rank reads its own disjoint
    block of ``total_rows // world_size`` rows onto its local GPU. Equal row
    counts give every rank the same number of batches; with unequal counts
    the rank that finishes its epoch first leaves the others blocked in the
    gradient all-reduce.

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

        import pyarrow.parquet as pq

        rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1
        sizes = [pq.read_metadata(f).num_rows for f in files]
        per_rank = sum(sizes) // world_size
        if per_rank == 0:
            raise RuntimeError(
                f"{sum(sizes)} rows in {self.parquet_path} cannot be split "
                f"across {world_size} ranks"
            )

        # Read this rank's row block [lo, hi) from the files it overlaps
        # (the remainder of fewer than world_size rows is not used).
        lo, hi = rank * per_rank, (rank + 1) * per_rank
        parts, start = [], 0
        for f, n in zip(files, sizes):
            if start < hi and start + n > lo:
                part = cudf.read_parquet(f, columns=["center", "context"])
                parts.append(part.iloc[max(lo - start, 0):min(hi - start, n)])
            start += n
        df = cudf.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        self.center = torch.utils.dlpack.from_dlpack(
            df["center"].to_dlpack()
        ).contiguous()
        self.context = torch.utils.dlpack.from_dlpack(
            df["context"].to_dlpack()
        ).contiguous()

    def train_dataloader(self):
        return _gpu_batch_loader((self.center, self.context), self.batch_size)


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

    def train_dataloader(self):
        # (Batch,), (Batch,), (Batch,)
        return _gpu_batch_loader((self.center, self.context, self.distance), self.batch_size)


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

    def train_dataloader(self):
        # (Batch, Win), (Batch, Win), (Batch,)
        return _gpu_batch_loader(
            (self.context, self.context_distance, self.center), self.batch_size
        )


# --------------------------------------------------------------------------- #
# Walks as sequences: the skip-gram pairs are built on the GPU per batch
# (`SkipGram._expand_skipgram`), so only the walks are held, not their pairs.
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
                 shuffle=True, seed=0, rank=0, world_size=1):
        super().__init__()
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
        my = [f for k, f in enumerate(self.files) if k % n_w == w_id]
        for f_idx, f in enumerate(my):
            pf = pq.ParquetFile(f)
            for c_idx, b in enumerate(
                pf.iter_batches(columns=["vertices", "predicates"], batch_size=self.read_chunk_size)
            ):
                v = _flat_list_array_to_2d(b.column("vertices"))
                p = _flat_list_array_to_2d(b.column("predicates"))
                if v.shape[0] == 0:
                    continue
                v, p = self._maybe_shuffle(v, p, shard_id=w_id, f_idx=f_idx, c_idx=c_idx)
                for s in range(0, v.shape[0], eb):
                    yield v[s:s + eb], p[s:s + eb]

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
    """GPU-resident sibling of `_WalkSeqIterable`: turns the cuDF list columns
    into 2-D GPU int tensors once (DLPack, no copy off the GPU), then yields
    `emit_batch_size`-row `(vertices, predicates)` slices in a new random
    order each epoch. Single GPU only: the whole corpus sits on one GPU."""

    def __init__(self, cudf_df, *, emit_batch_size, shuffle=True, seed=0):
        super().__init__()
        for col in ("vertices", "predicates"):
            if col not in cudf_df.columns:
                raise ValueError(f"in-memory walks frame missing '{col}' list column")
        n = len(cudf_df)
        # Every walk has the same list length (cuGraph pads ended walks with
        # -1), so the flat leaves reshape to (n, length).
        self.vertices = self._as_matrix(cudf_df["vertices"], n)
        self.predicates = self._as_matrix(cudf_df["predicates"], n)
        self.emit_batch_size = int(emit_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = -1

    @staticmethod
    def _as_matrix(column, n):
        if n == 0:
            return torch.empty((0, 0), dtype=torch.int64, device="cuda")
        leaves = column.list.leaves
        if leaves.dtype.kind != "i":  # predicates can come back as float
            leaves = leaves.astype("int64")
        flat = torch.from_dlpack(leaves.to_dlpack()).contiguous()
        if flat.numel() % n:
            raise ValueError("walk lists do not all have the same length")
        return flat.view(n, -1)

    def __len__(self):
        return -(-self.vertices.shape[0] // self.emit_batch_size)

    def __iter__(self):
        self._epoch += 1
        n = self.vertices.shape[0]
        if self.shuffle and n:
            generator = torch.Generator(device=self.vertices.device)
            generator.manual_seed(self.seed + self._epoch)
            order = torch.randperm(n, generator=generator, device=self.vertices.device)
            vertices, predicates = self.vertices[order], self.predicates[order]
        else:
            vertices, predicates = self.vertices, self.predicates
        step = self.emit_batch_size
        for start in range(0, n, step):
            yield vertices[start : start + step], predicates[start : start + step]


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
