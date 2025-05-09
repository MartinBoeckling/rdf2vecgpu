from pathlib import Path
import argparse
import random
from itertools import groupby
from collections import defaultdict
import multiprocessing as mp
from gensim.models.word2vec import Word2Vec as W2V
from igraph import Graph
import numpy as np
from tqdm import tqdm
import cugraph
import cudf
from reader import kg_reader
from helper.functions import determine_optimal_chunksize, get_gpu_cluster


class rdf2vec:
    def __init__(self, data_path: str, distance: int, max_walks: str,
                 train: bool, chunksize: int, save_path: str, cpu_count: int,
                 walk_strategy, gpu_count):
        """_summary_

        Args:
            data_path (str): _description_
            distance (int): _description_
            max_walks (str): _description_
            train (bool): _description_
            chunksize (int): _description_
            save_path (str): _description_
            cpu_count (int): _description_
            walk_strategy (_type_): _description_
        """
        # transform string to Path structure
        self.data_path = Path(data_path)
        # assign distance variable to class variable
        self.distance = distance
        # assign maximum walks to class variable
        self.max_walks = max_walks
        # assign train to class variable
        self.train = train
        # assign chunksize to class variable
        self.chunksize = chunksize
        # assign savepath to class variable
        self.save_path = Path(save_path)
        # assign cpu count to class variable
        self.cpu_count = cpu_count
        # assign walk strategy to class variable
        self.walk_strategy = walk_strategy
        # create logging directory Path name based on file name
        logging_directory = self.save_path
        # create logging directory
        logging_directory.mkdir(parents=True, exist_ok=True)
        # extract all file paths from directory
        # create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        # assign GPU cluster if defined
        if gpu_count != 0:
            self.use_gpu = True
            self.gpu_cluster = get_gpu_cluster(gpu_count)
        else:
            self.use_gpu = False


    def graph_reader(self) -> Graph:
        graph_data = kg_reader.read_kg_file(self.data_path)
        print("Graph Summary:")
        print(graph_data.summary())
        return graph_data

    def _predicate_generation(self, path_list: str) -> list:
        """
        Generates a sequence of predicates for a given path from the knowledge graph.

        Args:
            path_list (str): The list of edges (path) for which to generate predicates.

        Returns:
            list: A list of predicates and nodes in the form of a sequence.
        """
        # assign class graph to graph variable
        # assign class graph to graph variable
        graph = self.graph
        # extract predicate of edge given edge id stored in numpy
        pred_values = [e.attributes()['predicate'] for e in graph.es(path_list)]
        # extract node sequences that are part of the edge path and flatten numpy array
        node_sequence = np.array([graph.vs().select(e.tuple).get_attribute_values(
            'name') for e in graph.es(path_list)]).flatten()
        # delete consecutive character values in numpy array based from prior matrix
        node_sequence = np.array([key for key, _group in groupby(node_sequence)]).tolist()
        # combine predicate values and node sequences to one single array
        if node_sequence:
            path_sequence = []
            for index, value in enumerate(node_sequence):
                node_label = value
                edge_label = pred_values[index]
                path_sequence.append(node_label)
                path_sequence.append(edge_label)
                if index >= len(pred_values) -1:
                    last_value = node_sequence[-1]
                    path_sequence.append(last_value)
                    break
        else:
            path_sequence = []
        # return path sequence numpy array
        return path_sequence

    def _random_walk_iteration(self, id_number: int) -> list:
        """
        Performs a random walk over the graph starting from the specified node ID.

        Args:
            id_number (int): The node ID from which to start the random walk.

        Returns:
            list: A list of walk sequences, each representing a series of nodes and predicates.
        """

        walk_iteration = 0
        walk_list = []
        for walk_iteration in self.max_walks:
            walk_iteration += 1
            walk_edges = self.graph.random_walk(start=id_number, steps=self.distance, return_type="edges")
            path_sequence = self._predicate_generation(walk_edges)
            walk_list.append(path_sequence)
        
        return walk_list

    def _bsf_walk_iteration(self, entity_id: int) -> list:
        # assign class graph variable to local graph variable
        graph = self.graph
        # assign class maxWalks variable to local maxWalks variable
        max_walks = self.max_walks
        # extract index of graph node
        node_index = graph.vs.find(entity_id).index
        # perform breadth-first search algorithm
        bfs_list = graph.bfsiter(node_index, 'out', advanced=True)
        # iterate over breadth-first search iterator object to filter those paths out
        # defined distance variable
        distance_list = [
            node_path for node_path in bfs_list if node_path[1] <= self.distance]
        # create vertex list from distance list extracting vertex element
        vertex_list = [vertex_element[0] for vertex_element in distance_list]
        # check if all paths should be extracted
        if max_walks == -1:
            pass
        else:
            # limit maximum walks to maximum length of walkSequence length
            vertex_list_len = len(vertex_list)
            if vertex_list_len < max_walks:
                max_walks = vertex_list_len
            # random sample defined maximumWalk from vertexList list
            random.seed(15)
            vertex_list = random.sample(vertex_list, max_walks)
        # compute shortest path from focused node index to extracted vertex list outputting edge ID
        shortest_path_list = graph.get_shortest_paths(
            v=node_index, to=vertex_list, output='epath')
        # extract walk sequences with edge id to generate predicates
        walk_sequence = list(map(self._predicate_generation, shortest_path_list))
        # return walkSequence list
        return walk_sequence

    def _gpu_corpus_construction(self, graph: Graph) -> list:
        edges_df = graph.get_edge_dataframe()
        edges_df = edges_df.rename(columns={"source": "source", "target": "destination"})
        vertex_index_list = [vertex.index for vertex in graph.vs]
        edges_cudf = cudf.DataFrame.from_pandas(edges_df)
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(edges_cudf, vertex_col_names=('source', 'destination'), property_columns=["predicate"])
        if self.walk_strategy == "random":
            vertex_index_list = vertex_index_list * self.max_walks
            walk_vertices = cudf.Series(vertex_index_list)
            random_walk_results, weights, walk_sizes = cugraph.random_walks(G, start_vertices=walk_vertices, max_depth=self.distance)
        elif self.walk_strategy == "bfs":
            walk_vertices = cudf.Series(vertex_index_list)
            bfs_df = cugraph.bfs(G, start_vertices=walk_vertices)
        else:
            raise NotImplementedError(f"No implementation of provided value for walk strategy: {self.walk_strategy}")
    
        return []
    

    def _cpu_corpus_construction(self, graph: Graph) -> list:
        """_summary_

        Args:
            graph (Graph): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            list: _description_
        """
        entities = [vertex.index for vertex in graph.vs]
        self.graph = graph
        if self.chunksize is None:
            chunksize = determine_optimal_chunksize(len(entities), self.cpu_count)
        else:
            chunksize = self.chunksize
        pool = mp.Pool(self.cpu_count)
        if self.walk_strategy == "bfs":
            walk_predicate_list = list(tqdm(pool.imap_unordered(self._bsf_walk_iteration, entities, chunksize=chunksize),
                                        desc='BFS Walk Extraction', total=len(entities), position=0, leave=True))
        elif self.walk_strategy == "random":
            walk_predicate_list = list(tqdm(pool.imap_unordered(self._random_walk_iteration, entities, chunksize=self.chunksize),
                                        desc='Random Walk Extraction', total=len(entities), position=0, leave=True))
        else:
            raise NotImplementedError(f"No implementation of provided value for walk strategy: {self.walk_strategy}")
        
        pool.close()
        corpus = [walk for entity_walks in walk_predicate_list for walk in entity_walks]
        return corpus

    def _transform(self, w2v_model: W2V, entity_data) -> defaultdict:
        dict_vect = {entity: w2v_model.wv.get_vector(entity) for entity in entity_data}
        vector_mapping = defaultdict(np.array, dict_vect)
        return vector_mapping

    def fit(self, graph: Graph) -> W2V:
        if self.use_gpu:
            self._gpu_corpus_construction(graph=graph)
        else:
            corpus = self._cpu_corpus_construction(graph)
            model = W2V(min_count=1, workers=self.cpu_count)
            model.build_vocab(corpus)
            model.train(corpus, total_examples=model.corpus_count, epochs=10)
        return model

    def fit_transform(self, graph: Graph) -> defaultdict:
        model = self.fit(graph)
        entity_data = graph.vs["name"]
        vector_representation = self._transform(model, entity_data)
        return vector_representation


if __name__ == '__main__':
# initialize the command line argparser
    parser = argparse.ArgumentParser(description='RDF2Vec argument parameters')
    # add train argument parser
    parser.add_argument('-t', '--train', default=False, action='store_true',
                        help="use parameter if Word2Vec training should be performed")
    # add path argument parser
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='string value to data path')
    # add distance argument parser
    parser.add_argument('-d', '--distance', type=int, required=True,
                        help='walk distance from selected node')
    # add walk number argument parser
    parser.add_argument('-w', '--walknumber', type=int, required=True,
                        help='maximum walk number from selected node')
    # add chunksize argument
    parser.add_argument('-chunk', '--chunksize', type=int, required=True,
                        help="use parameter to determine chunksize for parallel processing")
    parser.add_argument('-save', '--savepath', type=str, required=True,
                        help="use parameter to save path for files")
    parser.add_argument('-cpu', '--cpu_count', type=int, required=True,
                        help="number of CPU cores that are assigned to multiprocessing")
    parser.add_argument('-walk', '--walk_strategy', type=str, required=True,
                        choices=["random", "bfs"],
                        help="walk strategy for RDF2Vec")
    parser.add_argument('-gpu', '--gpu_count', type=int, required=False, default=0,
                        help="number of GPUs that are assigned for processing")
    # store parser arguments in args variable
    args = parser.parse_args()
    initiated_class = rdf2vec(data_path=args.path, distance=args.distance, max_walks=args.walknumber, train=args.train,
                chunksize=args.chunksize, save_path=args.savepath, cpu_count=args.cpu_count,
                walk_strategy=args.walk_strategy, gpu_count=args.gpu_count)
    
    kg_data = initiated_class.graph_reader()