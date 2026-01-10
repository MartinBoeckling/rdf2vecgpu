import lightning as L
from torch.utils.data import DataLoader, Dataset, RandomSampler, BatchSampler
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
