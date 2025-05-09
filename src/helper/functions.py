from loguru import logger
import cudf
from dask_cuda import LocalCUDACluster
import torch

def _generate_vocab(edge_df: cudf.DataFrame) -> tuple[cudf.Series, cudf.Series]:
    """_summary_

    Args:
        edge_df (cudf.DataFrame): _description_

    Returns:
        tuple[cudf.Series, cudf.Series]: _description_
    """
    vocabulary = cudf.concat([edge_df["subject"], edge_df["predicate"], edge_df["object"]], ignore_index=True).unique()
    tokeninzation, word = vocabulary.factorize()
    return tokeninzation, word

def determine_optimal_chunksize(length_iterable: int, cpu_count: int) -> int:
    """Method to determine optimal chunksize for parallelism of unordered method

    Args:
        length_iterable (int): Size of iterable

    Returns:
        int: determined chunksize
    """
    chunksize, extra = divmod(length_iterable, cpu_count * 4)
    if extra:
        chunksize += 1
    return chunksize


@logger.catch
def get_gpu_cluster() -> LocalCUDACluster:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        available_gpus = torch.cuda.device_count()
    else:
        raise ValueError("Cuda is not available, please check your installation or system configuration")
    gpu_cluster = LocalCUDACluster(n_workers= available_gpus,
                                   device_memory_limit=0.9,
                                   protocol="ucx",
                                   enable_tcp_over_ucx=True,
                                   enable_cudf_spill=True,
                                   enable_nvlink=True)
    logger.info(f"GPU cluster created with {available_gpus} workers")
    return gpu_cluster
