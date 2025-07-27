from loguru import logger
import cudf
import dask.dataframe as dd
import dask_cudf
from dask_cuda import LocalCUDACluster
import torch

def _generate_vocab(edge_df: cudf.DataFrame, multi_gpu: bool) -> tuple[cudf.Series, cudf.Series]:
    """Build a token ↔ string vocabulary from a triple DataFrame.

    The function flattens the three columns *(subject, predicate, object)*,
    removes duplicates, and returns two parallel cuDF -Series:

    * **tokenisation** – integer category codes (contiguous in ``[0, n)``)  
    * **word** – original string values (IRIs / literals)

    When *multi_gpu* is ``True`` the computation is performed with
    dask-cuDF—useful for datasets that exceed the memory of a single GPU.
    Otherwise, a plain cuDF workflow is used.

    Parameters
    ----------
    edge_df : cudf.DataFrame
        Triple table whose columns are named ``subject``, ``predicate``,
        ``object`` and contain **strings**.
    multi_gpu : bool
        If ``True`` run the unique/count/factorise steps on a Dask-CUDA
        cluster.

    Returns
    -------
    tuple[cudf.Series, cudf.Series]
        *(tokenisation, word)*, where both Series share the same length and
        index.  The first contains ``int32`` category IDs, the second the
        corresponding strings.

    Notes
    -----
    * For the single-GPU branch, the mapping is produced with
      :py:meth:`cudf.Series.factorize`, which guarantees deterministic,
      zero-based codes.
    * The Dask branch categorises the vocabulary to ensure identical codes
      across partitions before resetting the index.
    """
    if multi_gpu:
        vocabulary = dd.concat([edge_df["subject"], edge_df["predicate"], edge_df["object"]], axis=0, ignore_index=True).unique()
        vocabulary = vocabulary.drop_duplicates()
        vocabulary = vocabulary.to_frame()
        vocabulary_categories = vocabulary.categorize()
        vocabulary_categories = vocabulary_categories.reset_index()
        vocabulary_categories.columns = ["index", "word"]
        return vocabulary_categories["index"], vocabulary_categories["word"]

    else:
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

def get_gpu_cluster() -> LocalCUDACluster:
    """
    Spin up a local **Dask-CUDA** cluster with one worker per detected GPU.

    The function first checks whether CUDA is available via
    ``torch.cuda.is_available()``.  If no GPU is present, it raises a
    ``ValueError`` to prevent silent fallback to CPU execution.  Otherwise, it
    launches a :class:`dask_cuda.LocalCUDACluster` configured for high-throughput
   , peer-to-peer communication:

    * **n_workers** – equals the number of visible CUDA devices  
    * **device_memory_limit** – 90 % of each GPU’s memory  
    * **protocol="ucx"** – enables GPUDirect RDMA with TCP fallback  
    * **enable_cudf_spill=True** – spills cuDF partitions to host when memory
      pressure is high  
    * **enable_nvlink=True** – leverages NVLink pathways when available

    A log-level *INFO* message records the number of workers created.

    Returns
    -------
    dask_cuda.LocalCUDACluster
        Handle to the running cluster.  Pass it to
        :class:`dask.distributed.Client` for task submission.

    Raises
    ------
    ValueError
        If ``torch`` cannot detect any CUDA device on the host.

    Notes
    -----
    Remember to shut the cluster down with ``cluster.close()`` (or by closing
    the associated ``Client``) to free GPU resources when your workload
    completes.
    """
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