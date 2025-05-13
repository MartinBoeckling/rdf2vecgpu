import wandb
import torch
import gc
from pathlib import Path
import multiprocessing
import time
import random
from itertools import product
from src.reader.kg_reader import read_kg_file
from performance.evaluation_parameters import (EPOCHS, WALK_NUMBER, MAX_DEPTH, RANDOM_STATE, EVALUATION_RUNS, TIMEOUT, MIN_COUNT, WINDOW_SIZE, VECTOR_SIZE)
from src.gpu_rdf2vec import GPU_RDF2Vec

def perform_rdf2vec_gpu_run(path: str, epoch: int, walk_number: int, max_depth: int):

    random_seed = random.SystemRandom().randint(0, 2**31 - 1)
    gpu_rdf2vec_model = GPU_RDF2Vec(
        walk_strategy="random",
        walk_depth=max_depth,
        walk_number=walk_number,
        embedding_model="skipgram",
        epochs=epoch,
        batch_size=None,
        vector_size=VECTOR_SIZE,
        window_size=WINDOW_SIZE,
        min_count=MIN_COUNT,
        learning_rate=0.01,
        negative_samples=5,
        random_state=random_seed,
        reproducible=False,
        multi_gpu=False,
        generate_artifact=False,
        cpu_count=20
    )
    run = wandb.init(
        project="gpu_rdf2vec_hpt_comparison",
        config={
            "epochs": epoch,
            "walk_number": walk_number,
            "max_depth": max_depth,
            "random_state": random_seed,
            "cpu_count": gpu_rdf2vec_model.cpu_count,
            "package": "rdf2vec_gpu",
            "dataset": path
        }
    )
    edge_data = gpu_rdf2vec_model.load_data(path)
    embeddings = gpu_rdf2vec_model.fit_transform(edge_df=edge_data, walk_vertices=None)
    wandb.log({"status": "success"})
    wandb.finish(exit_code=0)
    # garbage collection
    del gpu_rdf2vec_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

if __name__ == "__main__":
    product_params = product(EPOCHS, WALK_NUMBER, MAX_DEPTH)
    product_params = list(product_params)
    for parameters in product_params:
        epoch, walk_numbers, max_depths = parameters
        for i in range(EVALUATION_RUNS):
            perform_rdf2vec_gpu_run(path="data/wikidata5m/wikidata5m_kg.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
            perform_rdf2vec_gpu_run(path="data/fb15k-237/Release/fb15k_kg.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)

            perform_rdf2vec_gpu_run(path="data/generated_graphs/barabasi_graph_100.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
            perform_rdf2vec_gpu_run(path="data/generated_graphs/barabasi_graph_1000.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
            perform_rdf2vec_gpu_run(path="data/generated_graphs/barabasi_graph_10000.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
            
            perform_rdf2vec_gpu_run(path="data/generated_graphs/erdos_renyi_graph_100.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
            perform_rdf2vec_gpu_run(path="data/generated_graphs/erdos_renyi_graph_1000.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
            perform_rdf2vec_gpu_run(path="data/generated_graphs/erdos_renyi_graph_10000.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
            
            perform_rdf2vec_gpu_run(path="data/generated_graphs/random_graph_100.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
            perform_rdf2vec_gpu_run(path="data/generated_graphs/random_graph_1000.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
            perform_rdf2vec_gpu_run(path="data/generated_graphs/random_graph_10000.parquet", epoch=epoch, walk_number=walk_numbers, max_depth=max_depths)
