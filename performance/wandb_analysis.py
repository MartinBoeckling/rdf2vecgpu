# import pandas as pd
import wandb
import pandas as pd

api = wandb.Api()

entity = "m_boeck"
project = "rdf2vec_runtime_comparison"
path = f"{entity}/{project}"

runs = api.runs(path)
print(runs)
run_list = []
for run in runs:
    run_name = run.name
    run_config = run.config
    run_summary = run.summary
    run_config["run_name"] = run_name
    run_config.update(run_summary)
    run_list.append(run_config)

data = pd.DataFrame(run_list)
data = data[data["status"] == "success"]
print(data)
run_dataset_group = data.groupby(["package", "dataset"]).agg(
    {
        "_runtime": ["min", "max", "mean", "std"],
        "run_name": "count"
    }
)

run_dataset_group.to_excel("wandb_runs.xlsx")
print(run_dataset_group)

# wandb_runs = pd.read_csv("wandb_runs.csv")
# print(wandb_runs)

# import random

# random_seed = random.SystemRandom().randint(0, 2**31 - 1)
# print(f"Random seed: {random_seed}")
