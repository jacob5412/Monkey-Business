import os
import mlflow
import pandas as pd

MLFLOW_ROOT_DIR = "SVM_Final_Models/SVM_Logs"
mlflow.set_tracking_uri(MLFLOW_ROOT_DIR)
client = mlflow.tracking.MlflowClient()


def get_experiment_ids(root_dir):
    experiments = []
    for dir_name in os.listdir(root_dir):
        experiment_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(experiment_path) and dir_name != ".trash":
            experiments.append(dir_name)
    return experiments


def get_run_ids(experiment_path):
    run_ids = []
    for run_id in os.listdir(experiment_path):
        run_path = os.path.join(experiment_path, run_id)
        if os.path.isdir(run_path) and run_id != ".trash":
            run_ids.append(run_id)
    return run_ids


experiments = get_experiment_ids(MLFLOW_ROOT_DIR)

data = {
    "Experiment ID": [],
    "Experiment Name": [],
    "Run ID": [],
    "Params": [],
    "Metrics": [],
}
param_keys = set()
metric_keys = set()

for experiment_id in experiments:
    experiment_path = os.path.join(MLFLOW_ROOT_DIR, experiment_id)
    experiment = client.get_experiment(experiment_id)
    experiment_name = experiment.name

    run_ids = get_run_ids(experiment_path)
    for run_id in run_ids:
        with mlflow.start_run(run_id=run_id):
            run_data = mlflow.active_run().data
            params = run_data.params
            metrics = run_data.metrics
            data["Experiment ID"].append(experiment_id)
            data["Experiment Name"].append(experiment_name)
            data["Run ID"].append(run_id)
            data["Params"].append(params)
            data["Metrics"].append(metrics)
            param_keys.update(params.keys())
            metric_keys.update(metrics.keys())

param_keys_list = list(param_keys)
metric_keys_list = list(metric_keys)

df = pd.DataFrame(data)
for key in param_keys_list:
    df[key] = df.apply(lambda row: row["Params"].get(key, None), axis=1)
for key in metric_keys_list:
    df[key] = df.apply(lambda row: row["Metrics"].get(key, None), axis=1)

df.drop(["Params", "Metrics"], axis=1, inplace=True)
csv_file_path = MLFLOW_ROOT_DIR + "/hyperparam_results.csv"
df.to_csv(csv_file_path, index=False)
print(f"CSV file saved at: {csv_file_path}")
