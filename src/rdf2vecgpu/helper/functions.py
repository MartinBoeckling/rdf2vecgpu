import cudf
import cupy
import dask
import dask.dataframe as dd
import dask_cudf
import torch
from torch.utils.dlpack import to_dlpack


def _assign_ids(partition, offset):
    """Per-partition int32 token assignment via `cupy.arange + offset`.

    Used as the per-partition body of the multi-GPU vocab build. Each partition
    of the deduplicated vocab gets a contiguous block of integer ids starting
    at `offset` (computed from the cumulative sum of preceding partition
    sizes), so token ids are globally unique and form a contiguous `[0, n)`
    range without any cross-partition shuffle.
    """
    return cudf.DataFrame(
        {
            "word": partition["word"].astype("string[pyarrow]"),
            "token": (cupy.arange(len(partition), dtype="int32") + int(offset)),
        }
    )


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
        # Build word2idx via hash-partition + per-partition arange.
        #
        # The previous approach was `concat(s, p, o).unique()` followed by
        # `vocabulary_df.categorize(columns=["word"])`, which funnels the full
        # deduplicated vocabulary onto a single worker for `sort_values()`. On
        # Wikidata-scale graphs (~390 M unique tokens) that single-worker sort
        # exceeds 40 GB of intermediate state and either OOMs the GPU or hangs
        # the worker. The replacement is embarrassingly parallel:
        #
        #   1. concat(s, p, o) into a single `word` column (still lazy)
        #   2. shuffle by `hash(word)` so identical strings co-locate on the
        #      same partition
        #   3. drop_duplicates() locally per partition — globally unique by
        #      construction
        #   4. persist so the follow-up `sizes.compute()` and per-partition id
        #      assignment see the *same* shuffle output (without persist, the
        #      shuffle re-rolls and partition assignments drift — we observed
        #      23 M → 15 M row loss between sizes.compute and the eventual
        #      write)
        #   5. compute partition sizes (a tiny cumsum payload) → per-partition
        #      offsets
        #   6. dask.delayed(_assign_ids) per partition: `cupy.arange + offset`
        #      gives globally-unique int32 token ids without any cross-
        #      partition shuffle
        n_hash_partitions = max(1, edge_df.npartitions)
        vocabulary_df = dd.concat(
            [edge_df["subject"], edge_df["predicate"], edge_df["object"]]
        ).to_frame(name="word")
        vocabulary_df = vocabulary_df.shuffle(
            on="word",
            npartitions=n_hash_partitions,
            shuffle_method="tasks",
        )
        vocabulary_df = vocabulary_df.drop_duplicates()
        # Persist so the follow-up `sizes.compute()` and per-partition id
        # assignment see the *same* shuffle output. The subsequent `compute()`
        # on partition sizes drains the persist, so an explicit
        # `distributed.wait` is unnecessary (and would require an active
        # Client, breaking unit tests that use the default synchronous
        # scheduler).
        vocabulary_df = vocabulary_df.persist()

        sizes = vocabulary_df.map_partitions(len).compute()
        offsets = [0]
        for s in sizes[:-1]:
            offsets.append(offsets[-1] + int(s))

        word2idx_meta = cudf.DataFrame(
            {
                "word": cudf.Series([], dtype="string[pyarrow]"),
                "token": cudf.Series([], dtype="int32"),
            }
        )
        delayed_parts = [
            dask.delayed(_assign_ids)(
                vocabulary_df.get_partition(i), offsets[i]
            )
            for i in range(vocabulary_df.npartitions)
        ]
        word2idx = dask_cudf.from_delayed(delayed_parts, meta=word2idx_meta)

        # Encode the three edge columns by broadcast-joining word2idx into the
        # edge table. The previous code used three plain `.merge()` calls
        # without `broadcast=True`, which default to a hash-shuffle join; with
        # a small word2idx partition count, the 1.38 B-row edge side gets
        # funneled through one worker (we observed 6+ hours at 100 % on a
        # single GPU with no progress). `broadcast=True` replicates the small
        # word2idx to every worker, so each edge partition does a local hash
        # join — no shuffle of the large side, one hash-table build per worker
        # per merge instead of one per partition.
        w2i_s = word2idx.rename(columns={"word": "subject", "token": "s_tok"})
        w2i_p = word2idx.rename(columns={"word": "predicate", "token": "p_tok"})
        w2i_o = word2idx.rename(columns={"word": "object", "token": "o_tok"})
        edge_df = (
            edge_df
            .merge(w2i_s, on="subject", broadcast=True)
            .merge(w2i_p, on="predicate", broadcast=True)
            .merge(w2i_o, on="object", broadcast=True)
        )
        edge_df = edge_df[["s_tok", "p_tok", "o_tok"]].rename(
            columns={"s_tok": "subject", "p_tok": "predicate", "o_tok": "object"}
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
