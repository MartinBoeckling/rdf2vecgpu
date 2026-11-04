"""
Unit tests for the `read_kg_file` helper.

Assumptions
-----------
*   The tests use **pytest** plus the standard library only.  No external test
    doubles or heavy I/O dependencies are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import pytest
from src.rdf2vecgpu.reader.kg_file_reader import KGFileReader
import cudf


def _sample_edges() -> pd.DataFrame:
    """Return a tiny DataFrame representing two triples."""
    return cudf.DataFrame(
        {
            "subject": ["s1", "s2"],
            "predicate": ["p1", "p2"],
            "object": ["o1", "o2"],
        }
    )


@pytest.mark.parametrize("ext", [".csv"])
def test_read_tabular_files(tmp_path: Path, ext: str) -> None:
    """The loader should parse CSV or TSV into the expected record order."""
    df = _sample_edges()
    fp = tmp_path / f"kg{ext}"

    if ext == ".csv":
        df.to_csv(fp, index=False)
    kg_reader = KGFileReader(file_path=fp, multi_gpu=False)
    result = kg_reader.read()
    expected = df[["subject", "predicate", "object"]]
    assert result.equals(expected)


def test_read_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = _sample_edges()

    # Patch pd.read_parquet so we don't need an actual parquet file or pyarrow.
    monkeypatch.setattr(pd, "read_parquet", lambda _: df)

    fp = tmp_path / "kg.parquet"
    df.to_parquet(fp, index=False)
    kg_reader = KGFileReader(file_path=fp, multi_gpu=False)
    result = kg_reader.read()
    expected = df[["subject", "predicate", "object"]]
    assert result.equals(expected)


def test_read_orc_single_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = _sample_edges()

    # Patch pd.read_orc so we don't need an actual orc file or pyarrow.
    monkeypatch.setattr(pd, "read_orc", lambda _: df)

    fp = tmp_path / "kg.orc"
    df.to_orc(fp, index=False)
    fp.touch()  # file must exist for Path checks inside the loader (if any)
    kg_reader = KGFileReader(file_path=fp, multi_gpu=False)
    result = kg_reader.read()
    expected = df[["subject", "predicate", "object"]]
    assert result.equals(expected)


# def test_hdf_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
#    df = _sample_edges()
#    monkeypatch.setattr(pd, "read_hdf", lambda _: df)
#
#    fp = tmp_path / "kg.hdf"
#    df.to_hdf(fp, key="data", index=False)
#    kg_reader = KGFileReader(file_path=fp, multi_gpu=False)
#    result = kg_reader.read()
#    print(result)
#    expected = df[["subject", "predicate", "object"]]
#    assert result.equals(expected)
#


def test_rdf_lib_reader_single_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the RDFlib branch without depending on the real library."""

    # Local import to expose module namespace for monkeypatching
    import src.rdf2vecgpu.reader.kg_file_reader as m  # type: ignore  # noqa: WPS433

    triples: List[Tuple[str, str, str]] = [
        ("s1", "p1", "o1"),
        ("s2", "p2", "o2"),
    ]

    # 1. Force the RDF code path
    monkeypatch.setattr(m, "guess_format", lambda _p: "turtle")

    # 2. Provide a stub Graph implementation that behaves like an iterable
    class DummyGraph(list):
        def parse(self, _p):  # noqa: D401
            self.extend(triples)

        def close(self):  # noqa: D401
            pass

    monkeypatch.setattr(m, "rdfGraph", DummyGraph)

    ttl_path = tmp_path / "kg.ttl"
    ttl_path.write_text("")  # content irrelevant for stub

    reader_kg = KGFileReader(ttl_path, multi_gpu=False)
    result = reader_kg.read()
    expected_df = cudf.DataFrame(triples, columns=["subject", "predicate", "object"])

    assert result.equals(expected_df)


def test_unknown_extension_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "kg.unknown"
    bad_path.touch()

    with pytest.raises(NotImplementedError):
        kg_reader = KGFileReader(bad_path, multi_gpu=False)
        kg_reader.read()

    with pytest.raises(NotImplementedError):
        kg_reader = KGFileReader(bad_path, multi_gpu=True)
        kg_reader.read()


def test_nt_file_single_gpu(tmp_path: Path) -> None:
    """The loader should parse N-Triples into the expected record order."""
    nt_content = '<http://a> <http://b> "Hello World" .'
    nt_content += "\n<http://c> <http://d> <http://e> ."
    p = tmp_path / "test_graph.nt"
    p.write_text(nt_content, encoding="utf-8")

    reader = KGFileReader(str(p), multi_gpu=False)
    df = reader.read()
    assert df.shape[0] == 2
    assert df.shape[1] == 3
    assert df["object"].iloc[0] == '"Hello World"'
    assert df["subject"].iloc[1] == "http://c"
    expected_records = cudf.DataFrame(
        {
            "subject": ["http://a", "http://c"],
            "predicate": ["http://b", "http://d"],
            "object": ['"Hello World"', "http://e"],
        }
    )
    assert df.equals(expected_records)


def test_read_parquet_with_col_map(tmp_path: Path) -> None:
    """`col_map` maps the file's column names to subject/predicate/object/weights."""
    df = cudf.DataFrame({"src": ["s1", "s2"], "rel": ["p1", "p2"], "dst": ["o1", "o2"], "extra": [0, 1]})
    fp = tmp_path / "kg.parquet"
    df.to_parquet(fp, index=False)
    reader = KGFileReader(file_path=fp, multi_gpu=False,
                          col_map={"src": "subject", "rel": "predicate", "dst": "object"})
    result = reader.read()
    assert list(result.columns) == ["subject", "predicate", "object"]
    assert result["subject"].to_arrow().to_pylist() == ["s1", "s2"]
    assert result["object"].to_arrow().to_pylist() == ["o1", "o2"]


def test_read_parquet_renames_weights_only(tmp_path: Path) -> None:
    df = cudf.DataFrame({"subject": ["s1"], "predicate": ["p1"], "object": ["o1"], "distance": [2.5]})
    fp = tmp_path / "kg.parquet"
    df.to_parquet(fp, index=False)
    reader = KGFileReader(file_path=fp, multi_gpu=False, col_map={"distance": "weights"}, walk_weighted=True)
    result = reader.read()
    assert list(result.columns) == ["subject", "predicate", "object", "weights"]
    assert result["weights"].values_host.tolist() == [2.5]


@pytest.mark.gpu
def test_nt_file_multi_gpu(tmp_path: Path) -> None:
    nt_content = '<http://a> <http://b> "Hello World" .'
    nt_content += "\n<http://c> <http://d> <http://e> .\n"
    p = tmp_path / "test_graph.nt"
    p.write_text(nt_content, encoding="utf-8")
    df = KGFileReader(str(p), multi_gpu=True).read().compute()
    df = df.dropna().sort_values("subject").reset_index(drop=True)
    expected = cudf.DataFrame(
        {
            "subject": ["http://a", "http://c"],
            "predicate": ["http://b", "http://d"],
            "object": ['"Hello World"', "http://e"],
        }
    )
    assert df.equals(expected)


def _nq_file(tmp_path: Path) -> Path:
    p = tmp_path / "graph.nq"
    p.write_text(
        "<http://a> <http://b> <http://c> <http://g> .\n"
        '<http://a> <http://b> "Hello World" <http://g> .\n'
        "<http://c> <http://d> <http://e> _:g1 .\n"
        "<http://c> <http://d> <http://f> .\n",
        encoding="utf-8",
    )
    return p


def test_nq_file_drops_graph_term(tmp_path: Path) -> None:
    df = KGFileReader(str(_nq_file(tmp_path)), multi_gpu=False).read()
    assert df["object"].to_arrow().to_pylist() == ["http://c", '"Hello World"', "http://e", "http://f"]
    assert df["subject"].to_arrow().to_pylist() == ["http://a", "http://a", "http://c", "http://c"]


@pytest.mark.gpu
def test_nq_file_multi_gpu(tmp_path: Path) -> None:
    df = KGFileReader(str(_nq_file(tmp_path)), multi_gpu=True).read().compute()
    assert sorted(df["object"].dropna().to_arrow().to_pylist()) == ['"Hello World"', "http://c", "http://e", "http://f"]


def test_read_parquet_rejects_columns_kwarg(tmp_path: Path) -> None:
    fp = tmp_path / "kg.parquet"
    _sample_edges().to_parquet(fp, index=False)
    reader = KGFileReader(file_path=fp, multi_gpu=False, read_kwargs={"columns": ["subject"]})
    with pytest.raises(ValueError, match="may not set 'columns'"):
        reader.read()
