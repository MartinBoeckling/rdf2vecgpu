import cudf
from cugraph import Graph, uniform_random_walks, bfs, filter_unreachable
from loguru import logger

class single_gpu_walk_corpus:
    def __init__(self, graph: Graph):
        self.G = graph

    @logger.catch
    def _replace_entities_with_tokens(self, tokeninzation: cudf.Series, word: cudf.Series, edge_df: cudf.DataFrame) -> tuple[cudf.DataFrame, cudf.DataFrame]:
        """_summary_

        Args:
            tokeninzation (cudf.Series): _description_
            word (cudf.Series): _description_
            edge_df (cudf.DataFrame): _description_

        Returns:
            tuple[cudf.DataFrame, cudf.DataFrame]: _description_
        """
        word2idx = cudf.concat([cudf.Series(tokeninzation), cudf.Series(word)], axis=1)
        word2idx.columns = ["token", "word"]
        edge_df["subject"] = edge_df.merge(word2idx, left_on="subject", right_on="word", how="left")["token"]
        edge_df["predicate"] = edge_df.merge(word2idx, left_on="predicate", right_on="word", how="left")["token"]
        edge_df["object"] = edge_df.merge(word2idx, left_on="object", right_on="word", how="left")["token"]
        edge_df = edge_df.astype("int32")
        return edge_df, word2idx

    @logger.catch
    def _triples_to_tokens(self, df: cudf.DataFrame, min_count) -> cudf.DataFrame:
        """_summary_

        Args:
            df (cudf.DataFrame): _description_

        Returns:
            cudf.DataFrame: _description_
        """
        df = df.sort_values(['walk_id', 'step'])

        # predicate tokens --------------------------------------------------------
        pred_tok = cudf.DataFrame({
            'walk_id': df.walk_id,
            'pos':     df.step * 2 + 1,
            'token':   df.predicate
        })

        # dst tokens --------------------------------------------------------------
        dst_tok = cudf.DataFrame({
            'walk_id': df.walk_id,
            'pos':     df.step * 2 + 2,
            'token':   df.dst
        })

        # src tokens (only for the first row in each walk) ------------------------
        first_rows = df[df.step == 0]
        src_tok = cudf.DataFrame({
            'walk_id': first_rows.walk_id,
            'pos':     [0] * len(first_rows),
            'token':   first_rows.src
        })

        # concat & sort  ----------------------------------------------------------
        tokens = cudf.concat([src_tok, pred_tok, dst_tok])
        token_counts = tokens.groupby('token').size()
        token_counts = token_counts[token_counts >= min_count]
        token_counts = token_counts.reset_index()
        tokens = tokens.loc[tokens["token"].isin(token_counts["token"])]
        return tokens.sort_values(['walk_id', 'pos'])

    @logger.catch
    def _skipgram_pairs(self, tokens: cudf.DataFrame, window: int):
        pairs = []

        # one inner merge per offset (all happen on-GPU)
        for d in range(-window, window + 1):
            if d == 0:
                continue
            left  = tokens.rename(columns={'token': 'center'})
            right = tokens.rename(columns={'token': 'context'})
            right = right.assign(pos=right.pos - d)      # shift

            pairs.append(
                left.merge(right, on=['walk_id', 'pos'],
                        how='inner')[['center', 'context']]
            )
        return cudf.concat(pairs, ignore_index=True)
    

    def _cbow_pairs(self, tokens: cudf.DataFrame, window: int):
        raise NotImplementedError("CBOW is not implemented yet")

    @logger.catch
    def bfs_walk(self, edge_df: cudf.DataFrame, walk_vertices: cudf.Series, walk_depth: int, random_state: int, word2vec_model: str, min_count: int) -> cudf.DataFrame:
        out = []
        max_walk_id = 0
        for v in walk_vertices.to_cupy():
            bfs_extraction = bfs(self.G, start_vertices=v, max_depth=walk_depth)
            bfs_extraction_filtered = filter_unreachable(bfs_extraction)
            bfs_edges = (bfs_extraction_filtered[bfs_extraction_filtered.predecessor != -1].rename(columns={"predecessor": "subject", "vertex": "object"}).reset_index(drop=True))
            outdeg = bfs_edges.groupby("subject").size().rename("out_deg")
            leave_intermediate = bfs_extraction_filtered[["vertex"]].merge(outdeg, left_on="vertex", right_index=True, how="left")
            leaves = leave_intermediate[leave_intermediate.out_deg.isnull()].rename(columns={"vertex": "object"}).reset_index(drop=True)
            walk_id_list = list(range(max_walk_id, len(leaves) + max_walk_id))
            leaves["walk_id"] = walk_id_list
            max_walk_id = max(walk_id_list) + 1
            walk_edges = cudf.DataFrame(columns=["subject", "object", "walk_id"])
            frontier = leaves[["object", "walk_id"]]
            while len(frontier):
                # join to find each frontier vertex’s parent
                step = frontier.merge(bfs_edges, on="object", how="left")   # adds `source`
                step = step.dropna(subset=["subject"])                   # reached the root?

                # collect the edges that belong to these walks
                walk_edges = cudf.concat([walk_edges, step[["subject", "object", "walk_id"]]],
                                        ignore_index=True)

                # next frontier is this layer’s parents
                frontier = step[["subject", "walk_id"]].rename(columns={"subject": "object"})
            walk_edges = (
                walk_edges.astype({"subject": "int32", "object": "int32", "walk_id": "int32"})
                        .sort_values(["walk_id", "subject"])
                        .reset_index(drop=True)
            )
            out.append(walk_edges)
        bfs_all = cudf.concat(out, ignore_index=True)
        bfs_all["step"] = bfs_all.groupby("walk_id").cumcount()
        bfs_all = bfs_all.merge(edge_df, left_on=["subject", "object"], right_on=["subject", "object"], how="left")[["subject", "predicate", "object", "walk_id", "step"]]
        triple_to_token_df = self._triples_to_tokens(bfs_all, min_count)
        if word2vec_model == "skipgram":
            skipgram_df = self._skipgram_pairs(triple_to_token_df, window=5)
            return skipgram_df
        elif word2vec_model == "cbow":
            cbow_df = self._cbow_pairs(triple_to_token_df, window=5)
            return cbow_df
        else:
            raise ValueError("word2vec_model should be either 'skipgram' or 'cbow'")

    @logger.catch
    def random_walk(self, edge_df: cudf.DataFrame, walk_vertices: cudf.Series, walk_depth: int, random_state: int, word2vec_model: str, min_count: int) -> cudf.DataFrame:
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
        random_walks, _, max_length = uniform_random_walks(self.G, start_vertices=walk_vertices, max_depth=walk_depth,random_state=random_state)
        group_keys = cudf.Series(range(len(random_walks))) // max_length
        transformed_random_walk = random_walks.to_frame(name="src")
        transformed_random_walk["walk_id"] = group_keys
        transformed_random_walk["dst"] = transformed_random_walk['src'].shift(-1)
        transformed_random_walk = transformed_random_walk.mask(transformed_random_walk == -1, [None, None, None]).dropna()

        transformed_random_walk["step"] = transformed_random_walk.groupby("walk_id").cumcount()
        merged_walks = transformed_random_walk.merge(edge_df, left_on=["src", "dst"], right_on= ["subject", "object"], how="left")[["src", "predicate", "dst", "walk_id", "step"]]
        merged_walks = merged_walks.dropna()
        # Implement mincount -> #TODO
        merged_walks.sort_values(["walk_id", "step"])
        triple_to_token_df = self._triples_to_tokens(merged_walks, min_count)
        if word2vec_model == "skipgram":
            skipgram_df = self._skipgram_pairs(triple_to_token_df, window=5)
            return skipgram_df
        elif word2vec_model == "cbow":
            cbow_df = self._cbow_pairs(triple_to_token_df, window=5)
            return cbow_df
        else:
            raise ValueError("word2vec_model should be either 'skipgram' or 'cbow'")

