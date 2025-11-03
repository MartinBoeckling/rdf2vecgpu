from loguru import logger
import cudf
import dask.dataframe as dd
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client


def _generate_vocab(
    edge_df: cudf.DataFrame, multi_gpu: bool
) -> tuple[cudf.Series, cudf.Series]:
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
        edge_df.index = edge_df.index.rename("row_id")
        edge_df = edge_df.reset_index()
        edge_melted = edge_df.melt(id_vars="row_id", var_name="role", value_name="word")
        vocabulary_categories = edge_melted.categorize(subset=["word"])
        vocabulary_categories["token"] = vocabulary_categories["word"].cat.codes
        vocabulary_categories["word"] = vocabulary_categories["word"].astype("string")
        return vocabulary_categories

    else:
        vocabulary = cudf.concat(
            [edge_df["subject"], edge_df["predicate"], edge_df["object"]],
            ignore_index=True,
        ).unique()
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
