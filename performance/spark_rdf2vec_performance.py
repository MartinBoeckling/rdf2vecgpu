from pyspark.sql import SparkSession
from sparkkgml.kg import KG
from sparkkgml.motif_walks import MotifWalks
from pathlib import Path
import multiprocessing
from performance.evaluation_parameters import (EPOCHS, WALK_NUMBER, MAX_DEPTH, RANDOM_STATE, EVALUATION_RUNS, TIMEOUT, MIN_COUNT, WINDOW_SIZE)
import wandb
from itertools import product
import random

def run_spark_rdf2vec(path: str, epochs: int, walk_number: int, max_depth: int, random_seed: int) -> None:

    spark = (SparkSession.
                builder.
                master("local[20]").
                appName("KG-Example").
                config("spark.driver.memory", "250g").
                config("spark.local.dir", "/ceph/mboeckli/rdf2vecgpu/spark_local").
                config("spark.sql.shuffle.partitions", "200").
                config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.12").
                getOrCreate())

    rdf_location = path

    file_path = Path(path)
    file_name = file_path.stem
    storage_path = Path(f"vector/spark_rdf2vec/{file_name}/embeddings_{epochs}_{walk_number}_{max_depth}")

    kg = KG(location=rdf_location, fmt='turtle', sparkSession=spark)

    graph_frame = kg.createKG()

    entities = list(kg.vertex_to_key_hashMap.keys())
    motif_walks = MotifWalks(kg_instance=kg, entities=entities, sparkSession=spark)

    paths_df = motif_walks.motif_walk(graph_frame, depth=max_depth, walktype='BFS')
    embeddings_df = motif_walks.word2Vec_embeddings(
        df=paths_df,
        vector_size=100,
        min_count=MIN_COUNT,
        num_partitions=20*4,
        step_size=0.025,
        max_iter=epochs,
        seed=random_seed,
        input_col="paths",
        output_col="vectors",
        window_size=WINDOW_SIZE,
        max_sentence_length=1000000
    )

    # # Display the embeddings
    embeddings_df.write.parquet(str(storage_path), mode="overwrite")

def spark_timeout_handler(path: str, epochs: int, walk_number: int, max_depth: int):
    result_queue = multiprocessing.Queue()
    cpu_number = 20
    random_seed = random.SystemRandom().randint(0, 2**31 - 1)
    wandb.init(
            project="rdf2vec_runtime_comparison",
            config={
                "epochs": epochs,
                "walk_number": walk_number,
                "max_depth": max_depth,
                "random_state": random_seed,
                "cpu_count": cpu_number,
                "dataset": path,
                "package": "spark_rdf2vec",
            }
        )
    def wrapper(queue, graph, epochs, walk_number, max_depth, random_seed):
        try:
            result = run_spark_rdf2vec(graph, epochs, walk_number, max_depth, random_seed)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    p = multiprocessing.Process(target=wrapper, args=(result_queue, path, epochs, walk_number, max_depth, random_seed))
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
        return result

if __name__ == "__main__":
    product_params = product(EPOCHS, WALK_NUMBER, MAX_DEPTH)
    product_params = list(product_params)

    for parameters in product_params:
        epoch, walk_numbers, max_depths = parameters
        for i in range(EVALUATION_RUNS):
            spark_timeout_handler(path="data/wikidata5m/wikidata5m_kg.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
            spark_timeout_handler(path="data/fb15k-237/Release/fb15k_kg.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)

            spark_timeout_handler("data/generated_graphs/barabasi_graph_100.ttl", epoch, walk_numbers, max_depths)
            spark_timeout_handler("data/generated_graphs/barabasi_graph_1000.ttl", epoch, walk_numbers, max_depths)
            spark_timeout_handler("data/generated_graphs/barabasi_graph_10000.ttl", epoch, walk_numbers, max_depths)

            spark_timeout_handler("data/generated_graphs/erdos_renyi_graph_100.ttl", epoch, walk_numbers, max_depths)
            spark_timeout_handler("data/generated_graphs/erdos_renyi_graph_1000.ttl", epoch, walk_numbers, max_depths)
            spark_timeout_handler("data/generated_graphs/erdos_renyi_graph_10000.ttl", epoch, walk_numbers, max_depths)

            spark_timeout_handler("data/generated_graphs/random_graph_100.ttl", epoch, walk_numbers, max_depths)
            spark_timeout_handler("data/generated_graphs/random_graph_1000.ttl", epoch, walk_numbers, max_depths)
            spark_timeout_handler("data/generated_graphs/random_graph_10000.ttl", epoch, walk_numbers, max_depths)
