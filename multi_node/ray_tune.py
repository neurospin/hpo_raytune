import os
import sys
import argparse
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import Tuner, RunConfig

# -------------------------------------------------------------------
# 1. Loading the data : a toy dataset (Breast Cancer)
# -------------------------------------------------------------------
def load_breast_cancer_from_sklearn():
    """
    Loads the beast cancer dataset from sklearn and 
    returns the X_train, X_val, y_train, y_val.
    """
    # Loading breast cancer dataset from sklearn
    data = load_breast_cancer()
    X_train, X_val, y_train, y_val = train_test_split(data.data, data.target, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val

# -------------------------------------------------------------------
# 2. Defining the model to train and evaluate
# -------------------------------------------------------------------
class SimpleMLP(nn.Module):
    """
    Simple Multi Layer Perceptron
    """
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

# -------------------------------------------------------------------
# 3. Training function : 1 per trial
# -------------------------------------------------------------------
def train_and_evaluate(config):
    """
    Function called for each Ray Tune trial.
    Params :
        - config : contains all the hyperparameters (ex: 'lr', 'batch_size'...)
    
    Returns the score : the metric provided in the yaml configuration file
    """

    # retrieve the context
    trial_id = tune.get_context().get_trial_id()

    # Loading the toy data 
    X_train, X_val, y_train, y_val = load_breast_cancer_from_sklearn()
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == y_batch).sum().item()
        
        accuracy = correct / len(val_dataset)
        epoch_loss = val_loss / len(val_loader)

        print(f"[Trial {trial_id}] Epoch {epoch+1:02d}/{config['epochs']} "
              f"| Loss: {epoch_loss:.4f} "
              f"| Accuracy: {accuracy:.4f}")
        
        # Send the metrics to Ray Tune
        #train.report({"loss": val_loss / len(val_loader), "accuracy": accuracy})
        tune.report({config["metric"]: accuracy})

# -------------------------------------------------------------------
# 3. Logique d'Exécution et de Restauration
# -------------------------------------------------------------------
if __name__ == "__main__":

    print("\n" + "="*80, flush=True)
    print("RAY TUNE HYPERPARAMETER OPTIMIZATION", flush=True)
    print("="*80, flush=True)

    parser  = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args    = parser.parse_args()

    # Opening the yaml configuration file
    print(f"Yaml configuration file : {args.config}", flush=True)
    with open(args.config, "r") as f:
        config_yaml = yaml.safe_load(f)

    # Initializing the cluster
    print("[Ray Tune] Initializing Ray cluster...", flush=True)
    ray.init(address="auto")
    print("[Ray Tune] Ray cluster initialized successfully!\n", flush=True)


    # Defining the searching algorithm OptunaSearch(can by modified) with its parameters
    metric  = config_yaml["search"]["metric"]
    mode    = config_yaml["search"]["mode"]
    algo    = OptunaSearch(metric=metric, mode=mode)

    # Defining the parameter search space : Retrieving all the configuration parameters
    search_space = {
        "lr"        : tune.loguniform(
            float(config_yaml["hyperparameters"]["lr_min"]), 
            float(config_yaml["hyperparameters"]["lr_max"])
        ),
        "batch_size": tune.choice(config_yaml["hyperparameters"]["batch_size"]),

        # Static params 
        "epochs"    : config_yaml["hyperparameters"]["epochs"],
        "metric"    : metric,
        "mode"      : mode

    }

    # Ray tune output directory
    resources       = config_yaml["tune"]["resources_per_trial"]
    num_samples     = config_yaml["tune"]["num_samples"]
    
    EXP_NAME        = config_yaml['tune']['experiment_name']
    #STORAGE_PATH    = os.path.abspath(f"./ray_results")
    STORAGE_PATH    = "/lustre/fsn1/projects/rech/name-project/jean-zay-id/path/to/ray_tune_exp_results"
    EXP_PATH        = os.path.join(STORAGE_PATH, EXP_NAME)

    print(f"[Ray Tune] Output  directory : {EXP_PATH}", flush=True)


    # Defining the scheduler / early stopping etc (can be modified)
    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2
    )

    # Defining the maximum concurrent per trials
    num_nodes = config_yaml["tune"]["max_concurrent_trials"]["num_nodes"]
    num_gpu_per_node = config_yaml["tune"]["max_concurrent_trials"]["num_gpu_per_node"]


    # ---------------------------------------------------------------
    # CREATE OR RESTORE THE EXPERIMENT
    # ---------------------------------------------------------------


    tune_config = tune.TuneConfig(
        metric=metric,
        mode=mode,
        scheduler=scheduler,
        search_alg=algo,
        num_samples=num_samples,
        max_concurrent_trials = num_nodes * num_gpu_per_node
    )
    
    # New Ray Tune experiment
    if not Tuner.can_restore(EXP_PATH):
        # A Ray Tune experiments already exists
        print(f"[Ray Tune] Starting new experiment at {EXP_PATH}", flush=True)

        # Defining the scheduler / early stopping etc (can be modified)
        scheduler = ASHAScheduler(
            grace_period=1,
            reduction_factor=2
        )

        run_config = RunConfig(
            name=EXP_NAME,
            storage_path=STORAGE_PATH,
        )

        tuner = Tuner(
            tune.with_resources(train_and_evaluate, resources),
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )
    else:
        
        print(f"[Ray Tune] Restoring previous experiment from {EXP_PATH}", flush=True)
        tuner = Tuner.restore(
            path=EXP_PATH,
            trainable=tune.with_resources(train_and_evaluate, resources),
            param_space=search_space, 
            resume_unfinished=True,
            resume_errored=True,
            restart_errored=True,
        )

    # ---------------------------------------------------------------
    # RUN
    # ---------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("STARTING OPTIMIZATION", flush=True)
    print("="*80 + "\n", flush=True)

    results = tuner.fit()

    # ---------------------------------------------------------------
    # ENDING : Checking if all the trials has been terminated
    # ---------------------------------------------------------------
    total_expected = tune_config.num_samples
    
    # Count the termined trials
    try:
        completed = len([r for r in results if r.metrics is not None and metric in r.metrics])
        print(f"[Ray Tune] Calculating number of trials correctly terminated : {completed}", flush=True)
    except:
        completed = results.num_terminated if hasattr(results, 'num_terminated') else 0
        print(f"[Ray Tune] Calculating number of other trials : {completed}", flush=True)
    
    print("\n" + "="*80, flush=True)
    print("OPTIMIZATION RESULTS", flush=True)
    print("="*80, flush=True)
    print(f"Total trials expected: {total_expected}", flush=True)
    print(f"Trials completed: {completed}", flush=True)
    print("="*80 + "\n", flush=True)

    # Final message : ask user to continue or not the experiment
    if completed < total_expected:
        print("[Ray Tune] Incomplete ray tune experiment → if you want to persue the experiment, realunch it.", flush=True)
        print(f"[Ray Tune] Reason: completed={completed}/{total_expected}", flush=True)
        print("\n" + "="*80, flush=True)
        sys.exit(12)
    else:
        print("\n" + "="*80, flush=True)
        print("[Ray Tune] All trials completed !", flush=True)
        best = results.get_best_result(metric=metric, mode=mode)
        print("\n" + "="*80, flush=True)
        print("BEST RESULTS", flush=True)
        print("="*80, flush=True)
        print(f"Best config: {best.config}", flush=True)
        print(f"Best {metric}: {best.metrics[metric]:.4f}", flush=True)
        print("="*80 + "\n", flush=True)
        sys.exit(0)