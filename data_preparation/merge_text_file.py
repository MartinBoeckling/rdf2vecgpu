from rdflib import Graph as RDFGraph, URIRef
import pandas as pd
from tqdm.auto import tqdm

def merge_graph_files(input_files: list, output_file: str, file_format: str) -> None:
    """_summary_

    Args:
        input_files (list): _description_
        output_file (str): _description_
        format (str): _description_

    Raises:
        ValueError: _description_
    """
    train_path, valid_path, test_path = input_files

    train_data = pd.read_csv(train_path, sep="\t", header=None)
    train_data.columns = ["subject", "predicate", "object"]

    validation_data = pd.read_csv(valid_path, sep="\t", header=None)
    validation_data.columns = ["subject", "predicate", "object"]

    test_data = pd.read_csv(test_path, sep="\t", header=None)
    test_data.columns = ["subject", "predicate", "object"]

    merged_triple = pd.concat([train_data, validation_data, test_data], axis=0)

    if file_format == "parquet":
        merged_triple.to_parquet(output_file)
    elif file_format == "ttl":
        g = RDFGraph()
        for _, row in tqdm(merged_triple.iterrows(), total=merged_triple.shape[0]):
            subj = URIRef(row['subject'])
            pred = URIRef(row['predicate'])
            obj = URIRef(row['object'])
            g.add((subj, pred, obj))
        
        g.serialize(destination=output_file, format='turtle')

    else:
        raise ValueError("Unsupported format. Use 'parquet' or 'ttl'.")

if __name__ == "__main__":
    file_list = ["data/wikidata5m/wikidata5m_transductive_train.txt",
                 "data/wikidata5m/wikidata5m_transductive_valid.txt",
                 "data/wikidata5m/wikidata5m_transductive_test.txt"]
    merge_graph_files(input_files=file_list, output_file="data/wikidata5m/wikidata5m_kg.ttl", file_format="ttl")
    file_list = ["data/fb15k-237/Release/train.txt",
                 "data/fb15k-237/Release/valid.txt",
                 "data/fb15k-237/Release/test.txt"]
    merge_graph_files(input_files=file_list, output_file="data/fb15k-237/fb15k_kg.ttl", file_format="ttl")
    merge_graph_files(input_files=file_list, output_file="data/wikidata5m/wikidata5m_kg.ttl", file_format="ttl")