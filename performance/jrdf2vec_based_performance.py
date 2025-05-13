from evaluation_parameters import EPOCHS, WALK_NUMBER, MAX_DEPTH, RANDOM_STATE, EVALUATION_RUNS, TIMEOUT, MIN_COUNT, WINDOW_SIZE, VECTOR_SIZE
from pathlib import Path
import random
from itertools import product
import subprocess
import wandb
import os
import signal


def run_jrdf2vec(path: str, epochs: int, walk_number: int, max_depth: int) -> None:
    """_summary_

    Args:
        path (str): _description_
        epochs (int): _description_
        walk_number (int): _description_
        max_depth (int): _description_
    """
    file_path = Path(path)
    file_name = file_path.stem
    storage_path = Path(f"vector/jrdf2vec/{file_name}/embeddings_{epochs}_{walk_number}_{max_depth}.txt")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_text("", encoding="utf-8")
    cpu_number = 20
    random_seed = random.SystemRandom().randint(0, 2**31 - 1)
    try:
        wandb.init(
            project="rdf2vec_runtime_comparison",
            config={
                "epochs": epochs,
                "walk_number": walk_number,
                "max_depth": max_depth,
                "random_state": random_seed,
                "cpu_count": cpu_number,
                "dataset": path,
                "package": "jrdf2vec",
            }
        )

        random.seed(random_seed)
        command = [
            "java",
            "-jar",
            "/ceph/mboeckli/rdf2vecgpu/jrdf2vec-1.3-SNAPSHOT.jar",
            "-graph",
            path,
            "-epochs",
            str(epochs),
            "-numberOfWalks",
            str(walk_number),
            "-depth",
            str(max_depth),
            "-threads",
            str(cpu_number),
            "-walkGenerationMode",
            "RANDOM_WALKS",
            "-embedText",
            "-minCount",
            str(MIN_COUNT),
            "-window",
            str(WINDOW_SIZE),
            "-dimension",
            str(VECTOR_SIZE),
            "-noVectorTextFileGeneration"
        ]
        print("Command to be executed:", " ".join(command))

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        stdout, stderr = process.communicate(timeout=TIMEOUT)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        wandb.log({"status": "success"})
        wandb.finish(exit_code=0)
        

    except subprocess.TimeoutExpired:
        wandb.alert(
            title="Walk Extraction Timeout",
            text=f"Timeout during extraction of graph: {path}",
            level=wandb.AlertLevel.WARN
        )
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        wandb.log({"status": "timeout"})
        wandb.finish(exit_code=-1)

    except Exception as e:
        wandb.log({"status": "failed", "error": str(e)})
        wandb.finish(exit_code=1)

if __name__ == "__main__":
    product_params = product(EPOCHS, WALK_NUMBER, MAX_DEPTH)
    product_params = list(product_params)
    for parameters in product_params:
        epoch, walk_numbers, max_depths = parameters
          for i in range(EVALUATION_RUNS):
              run_jrdf2vec("/ceph/mboeckli/rdf2vecgpu/data/wikidata5m/wikidata5m_kg.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
              run_jrdf2vec(path="data/fb15k-237/Release/fb15k_kg.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)

              run_jrdf2vec(path="/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/barabasi_graph_100.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
              run_jrdf2vec(path="/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/barabasi_graph_1000.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
              run_jrdf2vec(path="/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/barabasi_graph_10000.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)

              run_jrdf2vec(path="/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/erdos_renyi_graph_100.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
              run_jrdf2vec(path="/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/erdos_renyi_graph_1000.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
              run_jrdf2vec(path="/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/erdos_renyi_graph_10000.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)

              run_jrdf2vec(path="/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/random_graph_100.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
              run_jrdf2vec(path="/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/random_graph_1000.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
              run_jrdf2vec(path="/ceph/mboeckli/rdf2vecgpu/data/generated_graphs/random_graph_10000.ttl", epochs=epoch, walk_number=walk_numbers, max_depth=max_depths)
