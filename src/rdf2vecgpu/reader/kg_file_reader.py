from pathlib import Path
from typing import Union
import cudf
from loguru import logger
from rdflib.util import guess_format
from rdflib import Graph as rdfGraph
from tqdm.auto import tqdm
import pandas as pd

try:
    import dask
    import dask.dataframe as dd

    HAS_DASK = True
except ImportError:  # pragma: no cover - env dependent
    dask = None  # type: ignore
    dd = None  # type: ignore
    HAS_DASK = False

DataFrameLike = Union[cudf.DataFrame, "dd.DataFrame"]


class KGFileReader:
    def __init__(
        self,
        file_path: str,
        multi_gpu: bool,
        col_map: dict[str, str] | None = None,
        read_kwargs: dict | None = None,
    ):
        self.file_path = Path(file_path)
        self.multi_gpu = multi_gpu
        self.file_ending = self.file_path.suffix
        self.col_map = col_map or {
            "subject": "subject",
            "predicate": "predicate",
            "object": "object",
        }
        self.read_kwargs = read_kwargs or {}
        if self.multi_gpu:
            self._ensure_dask_cudf()
            dask.config.set({"dataframe.backend": "cudf"})

    def _ensure_dask_cudf(self) -> None:
        """
        Checks if dask cudf can be imported
        """
        if not HAS_DASK:
            raise ImportError(
                "multi_gpu=True requires Dask and dask.dataframe to be installed.\n"
                "Install the multi-GPU extras, e.g.: `pip install gpu-rdf2vec[multi-gpu]`"
            )

    def read(self) -> DataFrameLike:
        # Check sequence for file ending
        if self.file_ending == ".parquet":
            return self._parquet_reader()
        elif self.file_ending in [".nt", ".nq"]:
            return self._nt_reader()
        elif self.file_ending == ".csv":
            return self._csv_reader()
        elif self.file_ending == ".orc":
            return self._orc_reader()
        elif self.file_ending in [".hdf", ".hdf5"]:
            return self._hdf_reader()
        else:
            # Check if file format is parseable by rdflib
            rdf_format = guess_format(self.file_path)
            if rdf_format is not None:
                self._rdf_lib_reader()
            else:
                logger.error(
                    f"Parsing of file format {self.file_ending} is currently not supported."
                )
                raise NotImplementedError(
                    f"Parsing of file format {self.file_ending} is currently not supported."
                )

    def _parquet_reader(self) -> DataFrameLike:
        if self.multi_gpu:
            kg_data = dd.read_parquet(
                self.file_path,
                columns=[
                    self.col_map["subject"],
                    self.col_map["predicate"],
                    self.col_map["object"],
                ],
                **self.read_kwargs,
            )
        else:
            kg_data = cudf.read_parquet(
                self.file_path,
                columns=[
                    self.col_map["subject"],
                    self.col_map["predicate"],
                    self.col_map["object"],
                ],
                **self.read_kwargs,
            )

        return kg_data

    def _csv_reader(self) -> DataFrameLike:
        if self.multi_gpu:
            kg_data = dd.read_csv(
                self.file_path,
                **self.read_kwargs,
            )
        else:
            kg_data = cudf.read_csv(
                self.file_path,
                **self.read_kwargs,
            )

        return kg_data

    def _nt_reader(self) -> DataFrameLike:
        if self.multi_gpu:
            df = dd.read_text(self.file_path).to_dataframe()
            df.columns = ["raw"]
        else:
            df = cudf.read_text(self.file_path)
            df.columns = ["raw"]

        nt_pattern = r"^<([^>]+)>\s+<([^>]+)>\s+(.*)\s+\.\s*$"
        extracted = df["raw"].str.extract(nt_pattern)
        extracted.columns = ["subject", "predicate", "object"]

        extracted["object"] = extracted["object"].str.replace(r"^<|>$", "", regex=True)
        kg_data = extracted[["subject", "predicate", "object"]]
        return kg_data

    def _orc_reader(self) -> DataFrameLike:
        if self.multi_gpu:
            kg_data = dd.read_orc(
                self.file_path,
                **self.read_kwargs,
            )
        else:
            kg_data = cudf.read_orc(
                self.file_path,
                **self.read_kwargs,
            )
        return kg_data

    def _hdf_reader(self) -> DataFrameLike:
        if self.multi_gpu:
            edge_df = dd.read_hdf(
                self.file_path,
                **self.read_kwargs,
            )
        else:
            edge_df = cudf.read_hdf(
                self.file_path,
                **self.read_kwargs,
            )
        return edge_df

    def _rdf_lib_reader(self) -> DataFrameLike:
        kg = rdfGraph()
        kg.parse(self.file_path)
        kg.close()
        edge_list = [triple for triple in tqdm(kg, desc="Parsing RDF with rdflib")]
        pd_edge_df = pd.DataFrame(edge_list, columns=["subject", "predicate", "object"])
        if self.multi_gpu:
            edge_df = dask.dataframe.from_pandas(pd_edge_df)
        else:
            edge_df = cudf.from_pandas(pd_edge_df)
        return edge_df
