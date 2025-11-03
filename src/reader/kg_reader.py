from pathlib import Path
from rdflib.util import guess_format
from rdflib import Graph as rdfGraph, URIRef, Namespace, Node, Literal
import pandas as pd
from tqdm.auto import tqdm
import urllib.parse


def read_kg_file(file_path: str) -> tuple:
    """Warpper method to read different KG files in. If the edge dataframe uses a rdflib supported file type,
    rdflib is used. In case the KG is provided in a DeltaTable format, those readers it is used to read the file in.

    Args:
        file_path (Path): File path for knowledge graph file

    Returns:
        Graph: Return tuple structure containing triple structure using the
    """
    # check if variable dataPath is a directory or a file
    file_path = Path(file_path)
    rdf_file_format = guess_format(str(file_path))
    if rdf_file_format is not None:
        kg = rdfGraph()
        kg.parse(file_path)
        kg.close()
        # prepare knowledge graph
        edge_list = [triple for triple in kg]
        edge_df = pd.DataFrame(edge_list, columns=["subject", "predicate", "object"])
    else:
        if file_path.suffix == ".parquet":
            edge_df = pd.read_parquet(str(file_path))
        elif file_path.suffix == ".csv":
            edge_df = pd.read_csv(str(file_path))
        elif file_path.suffix == ".txt":
            edge_df = pd.read_csv(
                str(file_path), sep="\t", names=["subject", "predicate", "object"]
            )
        else:
            file_ending = file_path.suffix
            raise NotImplementedError(
                f"For provided file path {str(file_path)} the file format {file_ending} is not implemented for read"
            )
    edge_df = edge_df[["subject", "object", "predicate"]]
    edge_df = edge_df.to_records(index=False)
    # kg_graph = Graph().TupleList(edges=edge_df, directed=True, edge_attrs=["predicate"])
    # return prepared edge dataframe
    return edge_df


def triple_to_ttl(file_path: str, destination_path: str):
    kg_data = read_kg_file(file_path=file_path)
    rdf_graph = rdfGraph()

    ex_namespace = Namespace("http://example.org/")
    rdf_graph.bind("ex", ex_namespace)
    for subj, obj, pred in tqdm(kg_data):
        subj = urllib.parse.quote_plus(str(subj))
        pred = urllib.parse.quote_plus(str(pred))
        obj = urllib.parse.quote_plus(str(obj))
        subj = URIRef(ex_namespace + str(subj))
        pred = URIRef(ex_namespace + str(pred))
        obj = URIRef(ex_namespace + str(obj))
        rdf_graph.add((subj, pred, obj))

    rdf_graph.serialize(destination=destination_path, format="ttl")
    print(f"Serialized RDF graph to {file_path} in ttl format.")
