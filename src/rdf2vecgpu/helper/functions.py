import cudf
import dask.dataframe as dd
import torch
from torch.utils.dlpack import to_dlpack


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
        # construct word2idx
        vocabulary = dd.concat(
            [edge_df["subject"], edge_df["predicate"], edge_df["object"]]
        ).unique()
        vocabulary_df = vocabulary.to_frame(name="word")
        word2idx = vocabulary_df.categorize(columns=["word"])
        word2idx["token"] = word2idx["word"].cat.codes
        word2idx = word2idx.astype({"word": "string[pyarrow]"})
        edge_df = (
            edge_df.merge(word2idx, left_on="subject", right_on="word")
            .drop(["word", "subject"], axis=1)
            .rename(columns={"token": "subject"})
        )
        edge_df = (
            edge_df.merge(word2idx, left_on="predicate", right_on="word")
            .drop(["word", "predicate"], axis=1)
            .rename(columns={"token": "predicate"})
        )
        edge_df = (
            edge_df.merge(word2idx, left_on="object", right_on="word")
            .drop(["word", "object"], axis=1)
            .rename(columns={"token": "object"})
        )
        edge_df = edge_df.astype(
            {"subject": "int32", "predicate": "int32", "object": "int32"}
        )
        return edge_df, word2idx

    else:
        vocabulary = cudf.concat(
            [edge_df["subject"], edge_df["predicate"], edge_df["object"]],
            ignore_index=True,
        ).unique()
        tokenization, word = vocabulary.factorize()
        word2idx = cudf.concat([cudf.Series(tokenization), cudf.Series(word)], axis=1)
        word2idx.columns = ["token", "word"]
        # Build a word→token lookup map and apply to all three columns at once
        lookup = cudf.Series(
            word2idx["token"].values, index=word2idx["word"]
        )
        for col in ("subject", "predicate", "object"):
            edge_df[col] = edge_df[col].map(lookup)
        edge_df = edge_df.dropna(subset=["subject", "predicate", "object"]).astype(
            {"subject": "int32", "predicate": "int32", "object": "int32"}
        )
        return edge_df, word2idx


def cudf_to_torch_tensor(df, column_name: str):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    return torch.utils.dlpack.from_dlpack(df[column_name].to_dlpack()).contiguous()


def torch_to_cudf(torch_tensor, multi_gpu: bool):
    if multi_gpu:
        raise NotImplementedError(
            "Conversion from torch Tensor to cuDF DataFrame is not implemented for multi-GPU."
        )
    else:
        column_major_tensor = torch_tensor.t().contiguous().t()

        dlpack_capsule = to_dlpack(column_major_tensor)
        return cudf.from_dlpack(dlpack_capsule)
