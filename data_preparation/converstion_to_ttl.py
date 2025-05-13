from src.reader.kg_reader import triple_to_ttl

if __name__ == "__main__":
    graph_paths = [
        ("data/generated_graphs/barabasi_graph_100.parquet", "data/generated_graphs/barabasi_graph_100.ttl"),
        ("data/generated_graphs/barabasi_graph_1000.parquet", "data/generated_graphs/barabasi_graph_1000.ttl"),
        ("data/generated_graphs/barabasi_graph_10000.parquet", "data/generated_graphs/barabasi_graph_10000.ttl"),
        ("data/generated_graphs/erdos_renyi_graph_100.parquet", "data/generated_graphs/erdos_renyi_graph_100.ttl"),
        ("data/generated_graphs/erdos_renyi_graph_1000.parquet", "data/generated_graphs/erdos_renyi_graph_1000.ttl"),
        ("data/generated_graphs/erdos_renyi_graph_10000.parquet", "data/generated_graphs/erdos_renyi_graph_10000.ttl"),
        ("/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/random_graph_100.parquet", "/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/random_graph_100.ttl"),
        ("/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/random_graph_1000.parquet", "/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/random_graph_1000.ttl"),
        ("/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/random_graph_10000.parquet", "/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/random_graph_10000.ttl")
    ]

    for source_path, destination_path in graph_paths:
        triple_to_ttl(file_path=source_path, destination_path=destination_path)