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
from src.reader.kg_reader import read_kg_file

def _sample_edges() -> pd.DataFrame:
    """Return a tiny DataFrame representing two triples."""
    return pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "predicate": ["p1", "p2"],
            "object": ["o1", "o2"],
        }
    )

@pytest.mark.parametrize("ext", [".csv", ".txt"])
def test_read_tabular_files(tmp_path: Path, ext: str) -> None:
    """The loader should parse CSV or TSV into the expected record order."""
    df = _sample_edges()
    fp = tmp_path / f"kg{ext}"

    if ext == ".csv":
        df.to_csv(fp, index=False)
    else:  # ".txt"  (tab‑separated, no header)
        df.to_csv(fp, sep="\t", header=False, index=False)

    result = read_kg_file(fp)

    expected = df[["subject", "object", "predicate"]].to_records(index=False)
    assert list(result) == list(expected)

def test_read_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = _sample_edges()

    # Patch pd.read_parquet so we don't need an actual parquet file or pyarrow.
    monkeypatch.setattr(pd, "read_parquet", lambda _: df)

    fp = tmp_path / "kg.parquet"
    fp.touch()  # file must exist for Path checks inside the loader (if any)

    result = read_kg_file(fp)
    expected = df[["subject", "object", "predicate"]].to_records(index=False)
    assert list(result) == list(expected)


def test_read_rdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercise the RDFlib branch without depending on the real library."""

    # Local import to expose module namespace for monkeypatching
    import src.reader.kg_reader as m  # type: ignore  # noqa: WPS433

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

    result = m.read_kg_file(ttl_path)
    expected_df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
    expected = expected_df[["subject", "object", "predicate"]].to_records(index=False)

    assert list(result) == list(expected)



def test_unknown_extension_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "kg.unknown"
    bad_path.touch()

    with pytest.raises(NotImplementedError):
        read_kg_file(bad_path)
