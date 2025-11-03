import cupy as cp
import cudf
import dask_cudf as dcudf
from loguru import logger
from cugraph import uniform_random_walks, filter_unreachable, bfs
import cugraph
from cugraph import Graph  # single-GPU Graph
from cugraph.dask import uniform_random_walks as dask_uniform_random_walks
from cugraph.dask.traversal.bfs import bfs as dask_bfs
import torch


class single_gpu_walk_corpus:
    """Build word2vec training pairs from a cuGraph single-GPU Graph.

    Generates Skip-gram or CBOW pairs from:
    - Uniform random walks
    - BFS paths from start vertices to leaves

    Tokens are integers derived directly from vertex/predicate ids, and an
    optional frequency threshold `min_count` filters rare tokens.
    """

    def __init__(self, graph: Graph, window_size: int):
        """Create a single-GPU corpus builder.

        Args:
            graph (Graph): cuGraph single-GPU Graph containing the data.
            window_size (int): Context window radius used to form pairs.
        """
        self.G = graph
        self.window_size = window_size

    def _replace_entities_with_tokens(
        self, tokeninzation: cudf.Series, word: cudf.Series, edge_df: cudf.DataFrame
    ) -> tuple[cudf.DataFrame, cudf.DataFrame]:
        """Map string/entity identifiers in edges to integer token ids.

        Args:
            tokeninzation (cudf.Series): Token ids aligned with `word`.
            word (cudf.Series): Original entity strings/ids aligned with tokens.
            edge_df (cudf.DataFrame): Edge list with columns ['subject','predicate','object'].

        Returns:
            tuple[cudf.DataFrame, cudf.DataFrame]:
                - Updated edge_df with integer tokens (int32).
                - word2idx DataFrame with columns ['token','word'].
        """
        word2idx = cudf.concat([cudf.Series(tokeninzation), cudf.Series(word)], axis=1)
        word2idx.columns = ["token", "word"]
        edge_df["subject"] = edge_df.merge(
            word2idx, left_on="subject", right_on="word", how="left"
        )["token"]
        edge_df["predicate"] = edge_df.merge(
            word2idx, left_on="predicate", right_on="word", how="left"
        )["token"]
        edge_df["object"] = edge_df.merge(
            word2idx, left_on="object", right_on="word", how="left"
        )["token"]
        edge_df = edge_df.astype("int32")
        return edge_df, word2idx

    def _triples_to_tokens(self, df: cudf.DataFrame, min_count) -> cudf.DataFrame:
        """Linearize (src, predicate, dst) triples into token sequences per walk.

        Expected input columns: ['src','predicate','dst','walk_id','step'].

        For each walk:
          pos=0 → src at step=0
          pos=2*k+1 → predicate at step=k
          pos=2*k+2 → dst at step=k

        Tokens with total frequency < min_count are removed.

        Args:
            df (cudf.DataFrame): Triple rows with walk and step metadata.
            min_count (int): Minimum token frequency to keep.
        """
        df = df.sort_values(["walk_id", "step"])

        # predicate tokens --------------------------------------------------------
        pred_tok = cudf.DataFrame(
            {"walk_id": df.walk_id, "pos": df.step * 2 + 1, "token": df.predicate}
        )

        # dst tokens --------------------------------------------------------------
        dst_tok = cudf.DataFrame(
            {"walk_id": df.walk_id, "pos": df.step * 2 + 2, "token": df.dst}
        )

        # src tokens (only for the first row in each walk) ------------------------
        first_rows = df[df.step == 0]
        src_tok = cudf.DataFrame(
            {
                "walk_id": first_rows.walk_id,
                "pos": [0] * len(first_rows),
                "token": first_rows.src,
            }
        )

        # concat & sort  ----------------------------------------------------------
        tokens = cudf.concat([src_tok, pred_tok, dst_tok])
        token_counts = tokens.groupby("token").size()
        token_counts = token_counts[token_counts >= min_count]
        token_counts = token_counts.reset_index()
        tokens = tokens.loc[tokens["token"].isin(token_counts["token"])]
        return tokens.sort_values(["walk_id", "pos"])

    def _skipgram_pairs(self, tokens: cudf.DataFrame, window: int):
        pairs = []

        # one inner merge per offset (all happen on-GPU)
        for d in range(-window, window + 1):
            if d == 0:
                continue
            left = tokens.rename(columns={"token": "center"})
            right = tokens.rename(columns={"token": "context"})
            right = right.assign(pos=right.pos - d)  # shift

            pairs.append(
                left.merge(right, on=["walk_id", "pos"], how="inner")[
                    ["center", "context"]
                ]
            )
        return cudf.concat(pairs, ignore_index=True)

    def _cbow_pairs(self, tokens: cudf.DataFrame, window: int):
        """
        For each position in each walk, collect all context tokens within the given window
        into a list, paired with the centre token.

        Returns a DataFrame with columns:
        - 'context' : list of context tokens
        - 'center'  : the centre token
        """
        pairs = []

        # for each offset, shift tokens to become a 'context' column
        for d in range(-window, window + 1):
            if d == 0:
                continue

            # rename token → context, and shift its pos so it lines up with the centre at pos+d
            ctx = tokens.rename(columns={"token": "context"}).assign(pos=tokens.pos - d)

            # join centre and this slice of context
            centre = tokens.rename(columns={"token": "center"})
            merged = centre.merge(ctx, on=["walk_id", "pos"], how="inner")[
                ["walk_id", "pos", "center", "context"]
            ]

            pairs.append(merged)

        # stack all (centre, single-context) rows
        all_pairs = cudf.concat(pairs, ignore_index=True)

        # aggregate each centre into a list of contexts
        cbow = (
            all_pairs.groupby(["walk_id", "pos", "center"])["context"]
            .agg(list)
            .reset_index()
        )
        return cbow[["context", "center"]]

    def bfs_walk(
        self,
        edge_df: cudf.DataFrame,
        walk_vertices: cudf.Series,
        walk_depth: int,
        random_state: int,
        word2vec_model: str,
        min_count: int,
    ) -> cudf.DataFrame:
        out = []
        max_walk_id = 0
        for v in walk_vertices.to_cupy():
            bfs_extraction = bfs(self.G, start=v, depth_limit=walk_depth)
            bfs_extraction_filtered = filter_unreachable(bfs_extraction)
            bfs_edges = (
                bfs_extraction_filtered[bfs_extraction_filtered.predecessor != -1]
                .rename(columns={"predecessor": "subject", "vertex": "object"})
                .reset_index(drop=True)
            )
            outdeg = bfs_edges.groupby("subject").size().rename("out_deg")
            leave_intermediate = bfs_extraction_filtered[["vertex"]].merge(
                outdeg, left_on="vertex", right_index=True, how="left"
            )
            leaves = (
                leave_intermediate[leave_intermediate.out_deg.isnull()]
                .rename(columns={"vertex": "object"})
                .reset_index(drop=True)
            )
            walk_id_list = list(range(max_walk_id, len(leaves) + max_walk_id))
            leaves["walk_id"] = walk_id_list
            max_walk_id = max(walk_id_list) + 1
            walk_edges = cudf.DataFrame(columns=["subject", "object", "walk_id"])
            frontier = leaves[["object", "walk_id"]]
            while len(frontier):
                # join to find each frontier vertex’s parent
                step = frontier.merge(
                    bfs_edges, on="object", how="left"
                )  # adds `source`
                step = step.dropna(subset=["subject"])  # reached the root?

                # collect the edges that belong to these walks
                walk_edges = cudf.concat(
                    [walk_edges, step[["subject", "object", "walk_id"]]],
                    ignore_index=True,
                )

                # next frontier is this layer’s parents
                frontier = step[["subject", "walk_id"]].rename(
                    columns={"subject": "object"}
                )
            walk_edges = (
                walk_edges.astype(
                    {"subject": "int32", "object": "int32", "walk_id": "int32"}
                )
                .sort_values(["walk_id", "subject"])
                .reset_index(drop=True)
            )
            out.append(walk_edges)
        bfs_all = cudf.concat(out, ignore_index=True)
        bfs_all["step"] = bfs_all.groupby("walk_id").cumcount()
        bfs_all = bfs_all.merge(
            edge_df,
            left_on=["subject", "object"],
            right_on=["subject", "object"],
            how="left",
        )[["subject", "predicate", "object", "walk_id", "step"]]
        bfs_all = bfs_all.rename(columns={"subject": "src", "object": "dst"})
        triple_to_token_df = self._triples_to_tokens(bfs_all, min_count)
        if word2vec_model == "skipgram":
            skipgram_df = self._skipgram_pairs(
                triple_to_token_df,
                window=self.window_size,
            )
            return skipgram_df
        elif word2vec_model == "cbow":
            cbow_df = self._cbow_pairs(
                triple_to_token_df,
                window=self.window_size,
            )
            return cbow_df
        else:
            raise ValueError("word2vec_model should be either 'skipgram' or 'cbow'")

    def random_walk(
        self,
        edge_df: cudf.DataFrame,
        walk_vertices: cudf.Series,
        walk_depth: int,
        random_state: int,
        word2vec_model: str,
        min_count: int,
    ) -> cudf.DataFrame:
        """_summary_

        Args:
            edge_df (cudf.DataFrame): _description_
            walk_vertices (cudf.Series): _description_

        Raises:
            NotImplementedError: _description_
            ValueError: _description_

        Returns:
            cudf.DataFrame: _description_
        """
        random_walks, _, max_length = uniform_random_walks(
            self.G,
            start_vertices=walk_vertices,
            max_depth=walk_depth,
            random_state=random_state,
        )
        group_keys = cudf.Series(range(len(random_walks))) // max_length
        transformed_random_walk = random_walks.to_frame(name="src")
        transformed_random_walk["walk_id"] = group_keys
        transformed_random_walk["dst"] = transformed_random_walk["src"].shift(-1)
        transformed_random_walk = transformed_random_walk.mask(
            transformed_random_walk == -1, [None, None, None]
        ).dropna()

        transformed_random_walk["step"] = transformed_random_walk.groupby(
            "walk_id"
        ).cumcount()
        merged_walks = transformed_random_walk.merge(
            edge_df, left_on=["src", "dst"], right_on=["subject", "object"], how="left"
        )[["src", "predicate", "dst", "walk_id", "step"]]
        merged_walks = merged_walks.dropna()
        merged_walks.sort_values(["walk_id", "step"])
        triple_to_token_df = self._triples_to_tokens(merged_walks, min_count)
        if word2vec_model == "skipgram":
            skipgram_df = self._skipgram_pairs(
                triple_to_token_df, window=self.window_size
            )
            return skipgram_df
        elif word2vec_model == "cbow":
            cbow_df = self._cbow_pairs(triple_to_token_df, window=self.window_size)
            return cbow_df
        else:
            raise ValueError("word2vec_model should be either 'skipgram' or 'cbow'")


def _ensure_dask_frame(df, nparts=None):
    """Return a dask_cudf.DataFrame (zero-copy if already dask)."""
    if isinstance(df, dcudf.DataFrame):
        return df
    return dcudf.from_cudf(df, npartitions=nparts or 1)


def _dst_per_partition(df: cudf.DataFrame):
    """Partition-safe dst = src.shift(-1) and drop last row."""
    df["dst"] = df["src"].shift(-1)
    return df.iloc[:-1]


class multi_gpu_walk_corpus:
    """
    A fully Dask/cuDF/cuGraph implementation of random-walk and BFS
    corpora that scales to many GPUs without triggering the 32-bit
    `size_type` overflow.
    """

    def __init__(self, graph: Graph, window_size: int):
        # Convert single-GPU Graphs to distributed Graphs on demand
        self.G = graph
        self.window_size = window_size

    # --------------- Low-level helpers (unchanged API) ------------------- #

    def _triples_to_tokens(
        self, df: dcudf.DataFrame, min_count: int
    ) -> dcudf.DataFrame:
        df = df.sort_values(["walk_id", "step"])

        # predicate tokens --------------------------------------------------
        pred_tok = dcudf.DataFrame(
            {"walk_id": df.walk_id, "pos": df.step * 2 + 1, "token": df.predicate}
        )

        # dst tokens --------------------------------------------------------
        dst_tok = dcudf.DataFrame(
            {"walk_id": df.walk_id, "pos": df.step * 2 + 2, "token": df.dst}
        )

        # src tokens (first row of each walk) ------------------------------
        first_rows = df[df.step == 0]
        src_tok = dcudf.DataFrame(
            {"walk_id": first_rows.walk_id, "pos": 0, "token": first_rows.src}
        )

        tokens = dcudf.concat([src_tok, pred_tok, dst_tok])

        # frequency filter
        tok_counts = tokens.groupby("token").size()
        tok_counts = tok_counts[tok_counts >= min_count].reset_index()
        tokens = tokens.merge(tok_counts[["token"]], on="token")

        return tokens.sort_values(["walk_id", "pos"])

    def _skipgram_pairs(self, tokens: dcudf.DataFrame, window: int):
        pairs = []
        for d in range(-window, window + 1):
            if d == 0:
                continue
            left = tokens.rename(columns={"token": "center"})
            right = tokens.rename(columns={"token": "context"})
            right = right.assign(pos=right.pos - d)

            pairs.append(
                left.merge(right, on=["walk_id", "pos"], how="inner")[
                    ["center", "context"]
                ]
            )
        return dcudf.concat(pairs, ignore_index=True)

    def _cbow_pairs(self, tokens: dcudf.DataFrame, window: int):
        """
        For CBOW we emit one pair per (target, context) exactly like Skip-Gram
        but invert the columns so downstream code can treat the left column as
        the *label* (word to predict) and the right column as a single context
        word.  If your trainer expects aggregated contexts, groupby/aggregate
        after `.compute()`.
        """
        pairs = []
        for d in range(-window, window + 1):
            if d == 0:
                continue
            left = tokens.rename(columns={"token": "target"})  # word to predict
            right = tokens.rename(columns={"token": "context"})
            right = right.assign(pos=right.pos - d)

            pairs.append(
                left.merge(right, on=["walk_id", "pos"], how="inner")[
                    ["target", "context"]
                ]
            )
        return dcudf.concat(pairs, ignore_index=True)

    # ---------------------- Random-walk corpus -------------------------- #

    def random_walk(
        self,
        edge_df: cudf.DataFrame | dcudf.DataFrame,
        walk_vertices: cudf.Series | dcudf.Series,
        walk_depth: int,
        random_state: int,
        word2vec_model: str,
        min_count: int,
    ):
        """Multi-GPU random-walk corpus builder (Skip-Gram / CBOW)."""

        # ---- Step 0 – inputs must be distributed ----
        edge_ddf = _ensure_dask_frame(edge_df)
        start_vertices = _ensure_dask_frame(cudf.DataFrame({"v": walk_vertices}))["v"]

        # ---- Step 1 – distributed random walks ----
        logger.info(
            "Running uniform_random_walks on {} GPUs …", dcudf.get_device_count()
        )
        walks_s, _, max_len = dask_uniform_random_walks(
            self.G,
            start_vertices=start_vertices,
            max_depth=walk_depth,
            random_state=random_state,
        )  # walks_s: dask_cudf.Series

        walks = walks_s.to_frame(name="src").reset_index(drop=False)
        walks = walks.rename(columns={"index": "global_pos"})

        # walk_id derived from the GLOBAL index – no overflow, no giant range
        walks["walk_id"] = walks["global_pos"].astype("int64") // max_len

        # ---- Step 2 – dst, step columns (partition-safe) ----
        walks = walks.map_partitions(_dst_per_partition)
        walks["step"] = walks.groupby("walk_id").cumcount()

        # ---- Step 3 – join predicates ------------------------
        merged = walks.merge(
            edge_ddf,
            left_on=["src", "dst"],
            right_on=["subject", "object"],
            how="left",
        ).dropna(subset=["predicate"])

        # keep needed columns only
        merged = merged[["src", "predicate", "dst", "walk_id", "step"]]

        # ---- Step 4 – tokens & (skip-gram | CBOW) pairs ----
        tokens = self._triples_to_tokens(merged, min_count)

        if word2vec_model == "skipgram":
            pairs = self._skipgram_pairs(
                tokens,
                window=self.window_size,
            )
        elif word2vec_model == "cbow":
            pairs = self._cbow_pairs(
                tokens,
                window=self.window_size,
            )
        else:
            raise ValueError("word2vec_model must be 'skipgram' or 'cbow'")

        return pairs  # still a dask_cudf.DataFrame – call .compute() when needed

    def bfs_walk(
        self,
        edge_df: cudf.DataFrame | dcudf.DataFrame,
        walk_vertices: cudf.Series | dcudf.Series,
        walk_depth: int,
        random_state: int,
        word2vec_model: str,
        min_count: int,
    ):
        """
        Distributed BFS-to-leaf walks.  The inner loop is identical to the
        single-GPU version but runs once **per worker** in parallel because
        each start vertex is scheduled on the GPU that owns it.
        """

        edge_ddf = _ensure_dask_frame(edge_df)
        start_vertices = _ensure_dask_frame(cudf.DataFrame({"v": walk_vertices}))["v"]

        # Helper that runs on a single GPU – we wrap the old single-GPU BFS
        def _bfs_from_vertex(v):
            bfs_extraction = cugraph.bfs(
                self.G.to_delayed(), start_vertices=v, max_depth=walk_depth
            )
            bfs_extraction_filtered = cugraph.filter_unreachable(bfs_extraction)

            bfs_edges = (
                bfs_extraction_filtered[bfs_extraction_filtered.predecessor != -1]
                .rename(columns={"predecessor": "subject", "vertex": "object"})
                .reset_index(drop=True)
            )

            outdeg = bfs_edges.groupby("subject").size().rename("out_deg")
            leaves = bfs_extraction_filtered[["vertex"]].merge(
                outdeg,
                left_on="vertex",
                right_index=True,
                how="left",
            )
            leaves = (
                leaves[leaves.out_deg.isnull()]
                .rename(columns={"vertex": "object"})
                .reset_index(drop=True)
            )

            # build walks root→leaf (same as before) ------------------
            walk_edges = cudf.DataFrame(
                columns=[
                    "subject",
                    "object",
                    "walk_id",
                ]
            )
            frontier = leaves.assign(
                walk_id=cp.arange(len(leaves), dtype="int32"),
            )

            while len(frontier):
                step = frontier.merge(bfs_edges, on="object", how="left")
                step = step.dropna(subset=["subject"])

                walk_edges = cudf.concat(
                    [
                        walk_edges,
                        step[["subject", "object", "walk_id"]],
                    ],
                    ignore_index=True,
                )

                frontier = step[["subject", "walk_id"]].rename(
                    columns={"subject": "object"}
                )

            walk_edges["step"] = walk_edges.groupby("walk_id").cumcount()
            return walk_edges

        # Apply the helper across the Dask Series of start vertices
        walks = start_vertices.map_partitions(
            lambda s: cudf.concat([_bfs_from_vertex(v) for v in s.to_cupy()]),
        )

        merged = walks.merge(
            edge_ddf,
            left_on=["subject", "object"],
            right_on=["subject", "object"],
            how="left",
        ).dropna(subset=["predicate"])

        merged = merged.rename(
            columns={
                "subject": "src",
                "object": "dst",
            },
        )
        tokens = self._triples_to_tokens(
            merged[
                [
                    "src",
                    "predicate",
                    "dst",
                    "walk_id",
                    "step",
                ]
            ],
            min_count,
        )

        if word2vec_model == "skipgram":
            pairs = self._skipgram_pairs(tokens, window=self.window_size)
        elif word2vec_model == "cbow":
            pairs = self._cbow_pairs(tokens, window=self.window_size)
        else:
            raise ValueError("word2vec_model must be 'skipgram' or 'cbow'")

        return pairs
