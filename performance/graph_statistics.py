from cugraph.centrality import betweenness_centrality, degree_centrality
from cugraph import Graph
import cudf
from pathlib import Path

def calculate_graph_statistics(path: str) -> None:
    gdf = cudf.read_parquet(path)
    print("-----"*10)
    print(Path(path).stem)
    G = Graph(directed=True)
    G.from_cudf_edgelist(gdf, source='subject', destination='object')

    degree_df = degree_centrality(G, normalized=False)
    mean_degree = degree_df["degree_centrality"].mean()
    print(f"Mean Degree Centrality: {mean_degree}")
    number_nodes = G.number_of_vertices()
    print(f"Number of Nodes: {number_nodes}")
    number_edges = G.number_of_edges()
    print(f"Number of Edges: {number_edges}")
    density = number_edges / (number_nodes * (number_nodes - 1))
    print(f"Density: {density}")
    betweenness_df = betweenness_centrality(G, normalized=False)
    mean_betweenness = betweenness_df["betweenness_centrality"].mean()
    print(f"Mean Betweenness Centrality: {mean_betweenness}")

if __name__ == "__main__":
    calculate_graph_statistics(path="data/wikidata5m/wikidata5m_kg.parquet")
    calculate_graph_statistics(path="data/fb15k-237/fb15k_kg.parquet")

    calculate_graph_statistics(path="data/generated_graphs/barabasi_graph_100.parquet")
    calculate_graph_statistics(path="data/generated_graphs/barabasi_graph_1000.parquet")
    calculate_graph_statistics(path="data/generated_graphs/barabasi_graph_10000.parquet")
    
    calculate_graph_statistics(path="data/generated_graphs/erdos_renyi_graph_100.parquet")
    calculate_graph_statistics(path="data/generated_graphs/erdos_renyi_graph_1000.parquet")
    calculate_graph_statistics(path="data/generated_graphs/erdos_renyi_graph_10000.parquet")
    
    calculate_graph_statistics(path="data/generated_graphs/random_graph_100.parquet")
    calculate_graph_statistics(path="data/generated_graphs/random_graph_1000.parquet")
    calculate_graph_statistics(path="data/generated_graphs/random_graph_10000.parquet")
    
