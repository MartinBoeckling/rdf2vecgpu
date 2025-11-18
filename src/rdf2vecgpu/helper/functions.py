import cudf
import dask.dataframe as dd
import torch


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
        subjects = vocabulary_categories[vocabulary_categories["role"] == "subject"][
            ["row_id", "token"]
        ]
        subjects = subjects.rename(columns={"token": "subject"})

        # Filter for predicates
        predicates = vocabulary_categories[
            vocabulary_categories["role"] == "predicate"
        ][["row_id", "token"]]
        predicates = predicates.rename(columns={"token": "predicate"})

        # Filter for objects
        objects = vocabulary_categories[vocabulary_categories["role"] == "object"][
            ["row_id", "token"]
        ]
        objects = objects.rename(columns={"token": "object"})

        # Merge them back together using the row_id
        subjects = subjects.set_index("row_id")
        predicates = predicates.set_index("row_id")
        objects = objects.set_index("row_id")
        kg_data = dd.concat([subjects, predicates, objects], axis=1).reset_index()
        kg_data = kg_data.astype(
            {"subject": "int64", "predicate": "int64", "object": "int64"}
        )
        word2idx = (
            vocabulary_categories[["token", "word"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return word2idx

    else:
        vocabulary = cudf.concat(
            [edge_df["subject"], edge_df["predicate"], edge_df["object"]],
            ignore_index=True,
        ).unique()
        tokenization, word = vocabulary.factorize()
        word2idx = cudf.concat([cudf.Series(tokenization), cudf.Series(word)], axis=1)
        word2idx.columns = ["token", "word"]
        return word2idx


def cudf_to_torch_tensor(df, column_name: str):
    return torch.utils.dlpack.from_dlpack(df[column_name].to_dlpack()).contiguous()


def torch_to_cudf(torch_tensor, multi_gpu: bool):
    if multi_gpu:
        raise NotImplementedError(
            "Conversion from torch Tensor to cuDF DataFrame is not implemented for multi-GPU."
        )
    else:
        cudf.from_dlpack(torch_tensor.to_dlpack())


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
