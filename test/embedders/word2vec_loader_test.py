"""`ParquetSkipGramDataModule` shards the pairs corpus by rows: every DDP rank
gets a disjoint block of the same size, whatever the file layout, so all ranks
run the same number of batches (unequal counts hang the gradient all-reduce)."""
from __future__ import annotations

from types import SimpleNamespace

import cudf
import pytest

from src.rdf2vecgpu.embedders.word2vec_loader import ParquetSkipGramDataModule


def _write_pairs(tmp_path, file_sizes):
    start = 0
    for i, n in enumerate(file_sizes):
        ids = list(range(start, start + n))
        cudf.DataFrame({"center": ids, "context": ids}).to_parquet(
            tmp_path / f"part.{i}.parquet", index=False
        )
        start += n
    return start


@pytest.mark.parametrize(
    "file_sizes, world_size",
    [([10, 3, 7, 1], 2), ([10, 3, 7, 1], 3), ([21], 4), ([1, 1, 1, 1, 1], 2)],
)
def test_ranks_get_equal_disjoint_row_blocks(tmp_path, file_sizes, world_size):
    total = _write_pairs(tmp_path, file_sizes)
    seen = []
    for rank in range(world_size):
        dm = ParquetSkipGramDataModule(str(tmp_path), batch_size=4)
        dm.trainer = SimpleNamespace(global_rank=rank, world_size=world_size)
        dm.setup()
        assert len(dm.center) == total // world_size
        assert len(dm.train_dataloader()) == -(-(total // world_size) // 4)
        seen.extend(dm.center.tolist())
    assert len(seen) == len(set(seen)) == (total // world_size) * world_size


def test_too_few_rows_for_ranks(tmp_path):
    _write_pairs(tmp_path, [1])
    dm = ParquetSkipGramDataModule(str(tmp_path), batch_size=4)
    dm.trainer = SimpleNamespace(global_rank=0, world_size=2)
    with pytest.raises(RuntimeError, match="cannot be split"):
        dm.setup()
