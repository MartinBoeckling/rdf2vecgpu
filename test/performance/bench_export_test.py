"""CPU tests for the KGrEAT export helpers in performance/bench_backends.py
(loaded from its file path: the harness is a script, not a package)."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BENCH = Path(__file__).resolve().parents[2] / "performance" / "bench_backends.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_backends_under_test", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_tsv(path):
    return pd.read_csv(path, sep="\t", header=None, index_col=0)


def test_write_kgreat_rows_l1_normalises_and_rounds(bench, tmp_path):
    ents = np.array(["http://x/a", "http://x/b", "http://x/zero"], dtype=object)
    M = np.array([[1.0, -3.0], [0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
    out = tmp_path / "e.tsv"
    with open(out, "w") as fh:
        n = bench._write_kgreat_rows(ents, M, fh)
    assert n == 3
    df = _read_tsv(out)
    assert df.index.tolist() == ents.tolist()
    np.testing.assert_allclose(df.to_numpy(), [[0.25, -0.75], [0.5, 0.5], [0.0, 0.0]])
    assert open(out).read().splitlines()[0] == "http://x/a\t0.25\t-0.75"


def test_write_kgreat_rows_applies_regex_and_list_filters(bench, tmp_path):
    import pyarrow as pa
    q1, p31, q7 = ("http://www.wikidata.org/entity/Q1", "http://www.wikidata.org/entity/P31",
                   "http://www.wikidata.org/entity/Q7")
    ents = pa.chunked_array([[q1, p31], [q7, "literal 42"]])
    M = np.arange(8, dtype=np.float32).reshape(4, 2) + 1
    regex = r"^http://www\.wikidata\.org/entity/Q"
    out = tmp_path / "f.tsv"
    with open(out, "w") as fh:
        n = bench._write_kgreat_rows(ents, M, fh, entity_regex=regex)
    assert n == 2 and _read_tsv(out).index.tolist() == [q1, q7]
    with open(out, "w") as fh:
        n = bench._write_kgreat_rows(ents, M, fh, entity_regex=regex, entity_set={q7, "literal 42"})
    assert n == 1 and _read_tsv(out).index.tolist() == [q7]
    with open(out, "w") as fh:
        assert bench._write_kgreat_rows(ents, M, fh, entity_set={"nope"}) == 0
    assert os.path.getsize(out) == 0


def test_load_entity_set_reads_one_iri_per_line(bench, tmp_path):
    p = tmp_path / "keep.txt"
    p.write_text("http://x/a\n\n  http://x/b  \n")
    assert bench._load_entity_set(str(p)) == {"http://x/a", "http://x/b"}
    assert bench._load_entity_set("") is None


def test_export_kgreat_tsv_frame_path_keeps_word_column_layout(bench, tmp_path):
    emb = pd.DataFrame({"token": [0, 1], "word": ["a", "b"],
                        "embedding_0": [2.0, 1.0], "embedding_1": [2.0, -1.0]})
    out = tmp_path / "sub" / "g.tsv"
    assert bench._export_kgreat_tsv(emb, str(out)) == 2
    np.testing.assert_allclose(_read_tsv(out).to_numpy(), [[0.5, 0.5], [0.5, -0.5]])
    assert bench._export_kgreat_tsv(emb, str(out), entity_regex="^b") == 1
    assert _read_tsv(out).index.tolist() == ["b"]


def test_export_kgreat_tsv_sharded_streams_chunks_through_the_model(bench, tmp_path):
    import pyarrow as pa

    class FakeModel:
        def iter_local_embeddings(self, *, chunk_rows):
            ents = ["http://x/Q1", "http://x/P2", "http://x/Q3", "http://x/Q4"]
            vecs = np.array([[1, 1], [2, 0], [0, 4], [3, -1]], dtype=np.float32)
            for i in range(0, 4, chunk_rows):
                yield (np.arange(i, min(i + chunk_rows, 4)),
                       pa.chunked_array([ents[i:i + chunk_rows]]), vecs[i:i + chunk_rows])

    out = tmp_path / "d" / "shard.tsv.rank0"
    written, total = bench._export_kgreat_tsv_sharded(
        FakeModel(), str(out), entity_regex="/Q", chunk_rows=3)
    assert (written, total) == (3, 4)
    df = _read_tsv(out)
    assert df.index.tolist() == ["http://x/Q1", "http://x/Q3", "http://x/Q4"]
    np.testing.assert_allclose(df.to_numpy(), [[0.5, 0.5], [0.0, 1.0], [0.75, -0.25]])


def test_concat_rank_shards_streams_in_rank_order_and_requires_all(bench, tmp_path):
    path = str(tmp_path / "all.tsv")
    for r, text in enumerate(["a\t1\n", "", "c\t3\nd\t4\n"]):
        with open(f"{path}.rank{r}", "w") as fh:
            fh.write(text)
    bench._concat_rank_shards(path, 3)
    assert open(path).read() == "a\t1\nc\t3\nd\t4\n"
    assert not any(os.path.exists(f"{path}.rank{r}") for r in range(3))
    with open(f"{path}.rank0", "w") as fh:
        fh.write("x\t0\n")
    with pytest.raises(RuntimeError, match="2/3 rank shards missing"):
        bench._concat_rank_shards(path, 3)
    assert os.path.exists(f"{path}.rank0")   # surviving shard is left for diagnosis
