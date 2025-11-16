from pathlib import Path
import dask_cudf
import cudf
from dask.bag import read_text


class KGFileReader:
    def __init__(
        self,
        file_path: str,
        multi_gpu: bool,
        col_map: dict[str, str] | None = None,
    ):
        self.file_path = Path(file_path)
        self.multi_gpu = multi_gpu
        self.file_ending = self.file_path.suffix
        self.col_map = col_map or {
            "subject": "subject",
            "predicate": "predicate",
            "object": "object",
        }

    def read(self):
        if self.file_ending == ".parquet":
            return self._parquet_reader()
        elif self.file_ending in [".nt", ".nq"]:
            return self._nt_reader()
        elif self.file_ending == ".csv":
            return self._csv_reader()

    def _parquet_reader(self) -> cudf.DataFrame | dask_cudf.DataFrame:
        if self.multi_gpu:
            kg_data = dask_cudf.read_parquet(
                self.file_path,
                columns=[
                    self.col_map["subject"],
                    self.col_map["predicate"],
                    self.col_map["object"],
                ],
            )
        else:
            kg_data = cudf.read_parquet(
                self.file_path,
                columns=[
                    self.col_map["subject"],
                    self.col_map["predicate"],
                    self.col_map["object"],
                ],
            )

        return kg_data

    def _csv_reader(self) -> cudf.DataFrame | dask_cudf.DataFrame:
        if self.multi_gpu:
            kg_data = dask_cudf.read_csv(
                self.file_path, usecols=["subject", "predicate", "object"]
            )
        else:
            kg_data = cudf.read_csv(
                self.file_path, usecols=["subject", "predicate", "object"]
            )

        return kg_data

    def txt_reader(self) -> cudf.DataFrame | dask_cudf.DataFrame:
        if self.multi_gpu:
            kg_data = dask_cudf.read_csv(
                self.file_path, sep="\t", usecols=["subject", "predicate", "object"]
            )
        else:
            kg_data = cudf.read_text(
                self.file_path, sep="\t", usecols=["subject", "predicate", "object"]
            )

        return kg_data

    def _nt_reader(self) -> cudf.DataFrame | dask_cudf.DataFrame:
        if self.multi_gpu:
            kg_data = dask_cudf.read_csv(
                self.file_path,
                sep=" ",
                names=["subject", "predicate", "object", "dot"],
                header=None,
            )
        else:
            kg_data = cudf.read_csv(
                self.file_path,
                sep=" ",
                names=["subject", "predicate", "object"],
                header=None,
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
