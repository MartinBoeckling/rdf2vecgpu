import glob
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
    Files are sharded across DataLoader workers (DDP-rank sharding is done by the
    caller, by trimming `.files`)."""

    def __init__(self, walks_dir, *, emit_batch_size, read_chunk_size=262_144,
                 shuffle=True, seed=0):
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
        # epoch counter so successive epochs (Lightning re-iterates the dataloader)
        # get a different shuffle. Bumped in __iter__.
        self._epoch = -1

    def __iter__(self):
        wi = get_worker_info()
        n_w, w_id = (wi.num_workers, wi.id) if wi else (1, 0)
        my = [f for k, f in enumerate(self.files) if k % n_w == w_id]
        eb = self.emit_batch_size
        self._epoch += 1
        for f_idx, f in enumerate(my):
            pf = pq.ParquetFile(f)
            for c_idx, b in enumerate(
                pf.iter_batches(columns=["vertices", "predicates"], batch_size=self.read_chunk_size)
            ):
                v = _flat_list_array_to_2d(b.column("vertices"))
                p = _flat_list_array_to_2d(b.column("predicates"))
                n = v.shape[0]
                if n == 0:
                    continue
                if self.shuffle:
                    # cuGraph emits the ~walk_number walks of each start vertex
                    # consecutively; without this shuffle a vertex's walks clump
                    # into one or two batches (over-aggregating its gradient /
                    # hitting the grad clip, then absent the rest of the epoch).
                    # Shuffling within the (large) read chunk spreads each
                    # vertex's walks across ~read_chunk_size/emit_batch_size
                    # batches. Seed mixes file index, chunk index, worker id and
                    # epoch so it's deterministic-per-run but varies across.
                    rng = np.random.default_rng(
                        (self.seed, self._epoch, w_id, f_idx, c_idx)
                    )
                    perm = rng.permutation(n)
                    v = v[perm]
                    p = p[perm]
                for s in range(0, n, eb):
                    yield v[s:s + eb], p[s:s + eb]


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
        ds = _WalkSeqIterable(
            self.walks_dir,
            emit_batch_size=self.batch_size,
            read_chunk_size=self.read_chunk_size,
            shuffle=self.shuffle,
            seed=self.seed,
        )
        if self.trainer is not None and self.trainer.world_size > 1:
            rank, ws = self.trainer.global_rank, self.trainer.world_size
            # File-level sharding: each rank takes every ws-th file.
            # When there are fewer files than ranks (e.g. a single-file corpus
            # from prepare_corpus / single-GPU walk-gen), fall back to giving
            # every rank all files so DDP can still proceed. The gradient
            # averaging across ranks is still correct (each rank computes the
            # same gradient; DDP averages → same update as 1-GPU). A
            # row-level stride would be more efficient but requires the
            # iterable to be rank-aware; the all-files fallback is the safe
            # default here and avoids a hard failure on small corpora.
            sharded = [f for i, f in enumerate(ds.files) if i % ws == rank]
            if sharded:
                ds.files = sharded
            # else: fewer files than ranks — all ranks read all files.
        self._iterable = ds

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
