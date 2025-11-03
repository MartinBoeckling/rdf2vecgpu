from pathlib import Path
import dask_cudf
import cudf
from dask.bag import read_text


class gpu_kg_file_reader:
    def __init__(self, file_path: str, multi_gpu: bool):
        self.file_path = Path(file_path)
        self.multi_gpu = multi_gpu
        self.file_ending = self.file_path.suffix

    def parquet_reader(self) -> cudf.DataFrame | dask_cudf.DataFrame:
        if self.multi_gpu:
            kg_data = dask_cudf.read_parquet(
                self.file_path, columns=["subject", "predicate", "object"]
            )
        else:
            kg_data = cudf.read_parquet(
                self.file_path, columns=["subject", "predicate", "object"]
            )

        return kg_data

    def csv_reader(self) -> cudf.DataFrame | dask_cudf.DataFrame:
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

    def nt_reader(self) -> cudf.DataFrame | dask_cudf.DataFrame:
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

    def ttl_reader(self) -> cudf.DataFrame | dask_cudf.DataFrame:
        kg_data = read_text(self.file_path)
        return kg_data
