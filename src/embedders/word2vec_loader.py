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

    def __init__(self, center_tensor: torch.Tensor, context_tensor: torch.Tensor, *, batch_size: int):
        super().__init__()
        assert center_tensor.device.type == "cuda", "tensors must be on GPU"
        assert center_tensor.shape == context_tensor.shape

        self.center  = center_tensor.contiguous()
        self.context = context_tensor.contiguous()
        self.batch_size = batch_size

        # length is reused in sampler
        self._dataset = _IndexDataset(len(self.center))

    def setup(self, stage: str | None = None):
        pass

    def train_dataloader(self):
        sampler = RandomSampler(self._dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)

        def _collate(indices: list[int]):  # indices comes from BatchSampler
            idx = torch.tensor(indices, device=self.center.device)
            return self.center[idx], self.context[idx]

        return DataLoader(
            self._dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate,
            num_workers=0,          # CUDA tensors + index dataset → no workers
            pin_memory=False,
        )