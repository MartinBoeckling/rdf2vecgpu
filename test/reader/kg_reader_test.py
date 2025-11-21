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
