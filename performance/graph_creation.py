from igraph import Graph
from tqdm import tqdm
import random
import pickle

def write_graph_parquet(kg_graph: Graph, file_name) -> None:

    kg_edge_df = kg_graph.get_edge_dataframe()
    kg_edge_df = kg_edge_df.rename({"source": "subject", "target": "object"}, axis=1)
    kg_edge_df = kg_edge_df.astype(dtype=str)
    kg_edge_df.to_parquet(f"data/generated_graphs/{file_name}.parquet")

number_vertex_list = [100, 1000, 10000]

with open("data/generated_graphs/relation.pkl", "rb") as file:
    relations = pickle.load(file)

for number_vertex in tqdm(number_vertex_list):
    generated_erdos_renyi_graph = Graph.Erdos_Renyi(n=number_vertex, directed=True, p=0.4)

    for edge in generated_erdos_renyi_graph.es:
        edge["predicate"] = random.choice(relations)
    write_graph_parquet(generated_erdos_renyi_graph, f"erdos_renyi_graph_{number_vertex}")

    generated_barabasi_graph = Graph.Barabasi(n=number_vertex, directed=True)

    for edge in generated_barabasi_graph.es:
        edge["predicate"] = random.choice(relations)

    write_graph_parquet(generated_barabasi_graph, f"barabasi_graph_{number_vertex}")

    generated_random_graph = Graph.Growing_Random(n=number_vertex, m=10)

    for edge in generated_random_graph.es:
        edge["predicate"] = random.choice(relations)


    write_graph_parquet(generated_random_graph, f"random_graph_{number_vertex}")
