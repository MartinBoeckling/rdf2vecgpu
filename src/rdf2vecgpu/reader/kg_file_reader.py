from pathlib import Path
from typing import Union
import cudf
from loguru import logger
from rdflib.util import guess_format
from rdflib import Graph as rdfGraph
from tqdm.auto import tqdm
import pandas as pd

DataFrameLike = Union["cudf.DataFrame", "dask.dataframe.DataFrame"]


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
            self._get_dask_cudf()

    def _get_dask_cudf(self) -> None:
        """
        Checks if dask cudf can be imported
        """
        try:
            import dask

            dask.config.set({"dataframe.backend": "cudf"})
        except ImportError:
            raise ImportError(
                "Dask needs to be installed for the run of the multi-gpu setup"
            )

    def read(self) -> cudf.DataFrame | dask.dataframe.DataFrame:
        # Check sequence for file ending
        if self.file_ending == ".parquet":
            return self._parquet_reader()
        elif self.file_ending in [".nt", ".nq"]:
            return self._nt_reader()
        elif self.file_ending == ".csv":
            return self._csv_reader()
        elif self.file_ending == ".orc":
            return self._orc_reader()
        elif self.file_ending in ["hdf", "hdf5"]:
            return self._hdf_reader()
        else:
            # Check if file format is parseable by rdflib
            if guess_format(self.file_ending):
                self._rdf_lib_reader()
            else:
                logger.error(
                    f"Parsing of file format {self.file_ending} is currently not supported."
                )
                raise NotImplementedError(
                    f"Parsing of file format {self.file_ending} is currently not supported."
                )

    def _parquet_reader(self) -> cudf.DataFrame | dask.dataframe.DataFrame:
        if self.multi_gpu:
            kg_data = dask.read_parquet(
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

    def _csv_reader(self) -> cudf.DataFrame | dask.dataframe.DataFrame:
        if self.multi_gpu:
            kg_data = dask.dataframe.read_csv(
                self.file_path,
                **self.read_kwargs,
            )
        else:
            kg_data = cudf.read_csv(
                self.file_path,
                **self.read_kwargs,
            )

        return kg_data

    def _nt_reader(self) -> cudf.DataFrame | dask.dataframe.DataFrame:
        if self.multi_gpu:
            kg_data = dask.dataframe.read_csv(
                self.file_path,
                sep=" ",
                names=["subject", "predicate", "object", "dot"],
                header=None,
                **self.read_kwargs,
            )
        else:
            kg_data = cudf.read_csv(
                self.file_path,
                sep=" ",
                names=["subject", "predicate", "object"],
                header=None,
                **self.read_kwargs,
            )

        kg_data = kg_data.drop(["dot"], axis=1)
        kg_data["subject"] = (
            kg_data["subject"].str.strip().str.replace("<", "").str.replace(">", "")
        )
        kg_data["predicate"] = (
            kg_data["predicate"].str.strip().str.replace("<", "").str.replace(">", "")
        )
        kg_data["object"] = (
            kg_data["object"].str.strip().str.replace("<", "").str.replace(">", "")
        )
        return kg_data

    def _orc_reader(self) -> cudf.DataFrame | dask.dataframe.DataFrame:
        if self.multi_gpu:
            kg_data = dask.dataframe.read_orc(
                self.file_path,
                **self.read_kwargs,
            )
        else:
            kg_data = cudf.read_orc(
                self.file_path,
                **self.read_kwargs,
            )
        return kg_data

    def _hdf_reader(self) -> cudf.DataFrame | dask.dataframe.DataFrame:
        if self.multi_gpu:
            dask.dataframe.read_hdf(
                self.file_path,
                **self.read_kwargs,
            )
        else:
            cudf.read_hdf(
                self.file_path,
                **self.read_kwargs,
            )

    def _rdf_lib_reader(self) -> cudf.DataFrame | dask.dataframe.DataFrame:
        kg = rdfGraph()
        kg.parse(self.file_path)
        kg.close()
        edge_list = [triple for triple in tqdm(kg)]
        pd_edge_df = pd.DataFrame(edge_list, columns=["subject", "predicate", "object"])
        if self.multi_gpu:
            edge_df = dask.dataframe.from_pandas(pd_edge_df)
        else:
            edge_df = cudf.from_pandas(pd_edge_df)
        return edge_df
