# import packages
import wandb
from pathlib import Path
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker
import pickle
import multiprocessing
import time
import random
from itertools import product
from src.reader.kg_reader import read_kg_file
from performance.evaluation_parameters import (EPOCHS, WALK_NUMBER, MAX_DEPTH, RANDOM_STATE, EVALUATION_RUNS, TIMEOUT, MIN_COUNT, WINDOW_SIZE, VECTOR_SIZE)


def run_pyrdf2vec(path: str, epochs: int, walk_number: int, max_depth: int) -> str:
    """_summary_

    Args:
        path (str): _description_
        epochs (int): _description_
        walk_number (int): _description_
        max_depth (int): _description_

    Returns:
        str: _description_
    """
    file_path = Path(path)
    file_name = file_path.stem
    kg_data = read_kg_file(file_path=path)

    knowledge_graph = KG()
    cpu_number = 20
    entity_set = set()
    for subject, object, predicate in kg_data:
        subject = str(subject)
        predicate = str(predicate)
        object = str(object)
        subj = Vertex((subject))
        obj = Vertex((object))
        entity_set.add(subject)
        entity_set.add(object)
        pred = Vertex((predicate), predicate=True, vprev=subj, vnext=obj)
        knowledge_graph.add_walk(subj, pred, obj)


    transformer = RDF2VecTransformer(
        Word2Vec(epochs=epochs, workers=cpu_number, window=WINDOW_SIZE, min_count=MIN_COUNT, vector_size=VECTOR_SIZE),
        walkers = [RandomWalker(
            max_walks=walk_number,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=cpu_number
        )],
        verbose=1
    )

    storage_path = Path(f"vector/pyrdf2vec/{file_name}/embeddings_{EPOCHS}_{WALK_NUMBER}_{MAX_DEPTH}.pkl")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    entities = list(entity_set)
    
    
    embeddings = transformer.fit_transform(knowledge_graph, entities)

    with open(storage_path, "wb") as f:
        pickle.dump(embeddings, f)

    return "done"


def monitor_run_pyrdf2vec(path: str, epochs: int, walk_number: int, max_depth: int) -> str:
    """_summary_

    Args:
        path (str): _description_
        epochs (int): _description_
        walk_number (int): _description_
        max_depth (int): _description_

    Raises:
        result: _description_

    Returns:
        str: _description_
    """
    result_queue = multiprocessing.Queue()
    cpu_number = 20
    def wrapper(queue, graph, epochs, walk_number, max_depth):
        try:
            result = run_pyrdf2vec(graph, epochs, walk_number, max_depth)
            queue.put(result)
        except Exception as e:
            queue.put(e)
    
    random_seed = random.SystemRandom().randint(0, 2**31 - 1)

    wandb.init(
        project="rdf2vec_runtime_comparison",
        config={
            "epochs": epochs,
            "walk_number": walk_number,
            "max_depth": max_depth,
            "random_state": random_seed,
            "cpu_count": cpu_number,
            "package": "pydf2vec",
            "dataset": path
        }
    )

    
    p = multiprocessing.Process(target=wrapper, args=(result_queue, path, epochs, walk_number, max_depth))

    p.start()
    p.join(timeout=TIMEOUT)
    if p.is_alive():
        print("Timeout reached. Killing process.")
        p.terminate()
        p.join()
        wandb.alert(
            title="Walk Extraction Timeout",
            text=f"Timeout during extraction of graph: {path}",
            level=wandb.AlertLevel.WARN
        )
        wandb.log({"status": "timeout"})
        wandb.finish(exit_code=1)
        return None  # or raise a TimeoutError
    else:
        result = result_queue.get()
        if isinstance(result, Exception):
            wandb.log({"status": "failed", "error": str(result)})
            wandb.finish(exit_code=1)
            raise result
        else:
            wandb.log({"status": "success"})
            wandb.finish(exit_code=0)
    
    result = run_pyrdf2vec(path, epochs, walk_number, max_depth)
    # wandb.log({"status": "success"})
    # wandb.finish(exit_code=0)
    return result


if __name__ == "__main__":
    product_params = product(EPOCHS, WALK_NUMBER, MAX_DEPTH)
    product_params = list(product_params)

    for parameters in product_params:
        epoch, walk_numbers, max_depths = parameters
        for i in range(EVALUATION_RUNS):
            monitor_run_pyrdf2vec(path="data/wikidata5m/wikidata5m_kg.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            monitor_run_pyrdf2vec(path="data/fb15k-237/Release/fb15k_kg.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)

            monitor_run_pyrdf2vec(path="data/generated_graphs/barabasi_graph_100.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            monitor_run_pyrdf2vec(path="data/generated_graphs/barabasi_graph_1000.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            monitor_run_pyrdf2vec(path="data/generated_graphs/barabasi_graph_10000.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            
            monitor_run_pyrdf2vec(path="data/generated_graphs/erdos_renyi_graph_100.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            monitor_run_pyrdf2vec(path="data/generated_graphs/erdos_renyi_graph_1000.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            monitor_run_pyrdf2vec(path="data/generated_graphs/erdos_renyi_graph_10000.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            
            monitor_run_pyrdf2vec(path="data/generated_graphs/random_graph_100.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            monitor_run_pyrdf2vec(path="data/generated_graphs/random_graph_1000.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            monitor_run_pyrdf2vec(path="data/generated_graphs/random_graph_10000.parquet", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
