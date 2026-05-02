# IntelliClave FL Process Flow

This file explains, step by step, how the current federated learning pipeline works in this repository.

It covers:
- what each FL file does
- how local model training works
- how the server gets initial weights
- how federated rounds happen
- what gets saved in `results/fl_rounds`

## 1. Big Picture

The current FL pipeline uses:
- `data/processed/client1.csv`
- `data/processed/client2.csv`
- `data/processed/client3.csv`

Each CSV represents one organization/client.

Each row has:
- `50` PCA features: `pca_0` to `pca_49`
- `1` label column: `label`

The label values in the CSVs are:
- `1 = WALKING`
- `2 = WALKING_UPSTAIRS`
- `3 = WALKING_DOWNSTAIRS`
- `4 = SITTING`
- `5 = STANDING`
- `6 = LAYING`

Inside the FL code, labels are converted from `1..6` to `0..5` because PyTorch `CrossEntropyLoss` expects zero-based class indices.

## 2. File-by-File Responsibilities

### `fl/data_utils.py`

This is the data entry point.

What it does:
- defines the class names
- resolves project paths
- finds the default client CSVs
- loads class weights from `data/class_weights.json`
- reads one client CSV
- validates that labels are exactly `1..6`
- splits the client CSV into local train/test sets
- standardizes features using `StandardScaler`
- wraps the split data into PyTorch `DataLoader`s

Important functions:
- `get_default_client_csvs()`
  Returns the default paths for `client1.csv`, `client2.csv`, and `client3.csv`.
- `load_class_weights()`
  Loads optional class weights for imbalanced multiclass training.
- `load_csv_data()`
  Reads a CSV, converts labels from `1..6` to `0..5`, performs train/test split, scales the features, and returns:
  - `train_loader`
  - `test_loader`
  - `metadata`

Where local data is prepared:
- local train/test splitting happens inside `load_csv_data()`
- this split is per client, not global

### `fl/model.py`

This defines the model architecture used everywhere.

What it does:
- defines `HARClassifier`
- builds a feed-forward neural network:
  - input layer for `50` PCA features
  - hidden layers: `128 -> 64`
  - output layer: `6` logits, one for each activity class

Important detail:
- there is no sigmoid or softmax layer at the end
- this is correct because `CrossEntropyLoss` expects raw logits

Used by:
- `train_local.py`
- `fl_client.py`
- `fl_server.py`
- `run_fl_simulation.py`
- `evaluate_global_model.py`

### `fl/train_local.py`

This is the standalone local training script for one client CSV.

What it does:
- loads one client CSV through `load_csv_data()`
- creates a model from `model.py`
- creates the loss function:
  - `CrossEntropyLoss`
  - optionally weighted with `data/class_weights.json`
- creates the optimizer:
  - `Adam`
- trains for `N` epochs on that one client only
- evaluates on that client's local test split
- optionally saves the model and training history

Important functions:
- `train_one_epoch()`
  Runs one local epoch over one client's train loader.
- `evaluate()`
  Computes:
  - accuracy
  - macro-F1
- `train_local()`
  Full local training workflow for one CSV.

Where local models are trained:
- local model training happens inside `train_local()`
- the actual weight updates happen in `train_one_epoch()`

Why this file matters:
- it is the sanity check before FL
- if `client1.csv`, `client2.csv`, and `client3.csv` all train correctly here, the FL pipeline is much easier to trust

### `fl/fl_client.py`

This file turns one client CSV into a Flower client.

What it does:
- defines `IntelliClaveClient`, a subclass of `fl.client.NumPyClient`
- loads that client's CSV
- creates the local model
- creates the local optimizer and loss
- receives global parameters from the server
- trains locally for a few epochs
- returns updated weights and metrics back to the server

Important methods:
- `get_parameters()`
  Returns the client's current model weights as NumPy arrays.
- `set_parameters()`
  Loads the server's global weights into the local client model.
- `fit()`
  Runs local training for one FL round and returns:
  - updated client weights
  - number of local training examples
  - local training metrics
- `evaluate()`
  Evaluates the received global model on that client's local test set

Where local FL training happens:
- during federated learning, local client-side training happens in `IntelliClaveClient.fit()`
- internally, that calls `train_one_epoch()` from `train_local.py`

### `fl/fl_server.py`

This is the FL server logic.

What it does:
- defines the custom FedAvg strategy `SaveModelStrategy`
- starts the Flower server
- aggregates client updates
- saves model checkpoints after each round
- saves round metrics to `results/fl_rounds/fl_metrics.json`

Important pieces:

#### `SaveModelStrategy`

Extends Flower's `FedAvg`.

It adds:
- saving raw aggregated weights as `.npz`
- saving PyTorch checkpoints as `.pth`
- writing per-round metrics to JSON

#### `aggregate_fit()`

Runs after client training updates arrive at the server.

What happens:
- Flower performs FedAvg aggregation
- the aggregated global weights are saved as:
  - `round_X.npz`
  - `global_model_round_X.pth`
  - `global_model_latest.pth`

#### `aggregate_evaluate()`

Runs after client evaluation results arrive.

What happens:
- Flower aggregates evaluation loss and metrics
- metrics are appended to `fl_metrics.json`

#### `weighted_average()`

Aggregates metrics like:
- `accuracy`
- `macro_f1`

using the number of examples from each client as weights.

### `fl/run_server.py`

This is just a small launcher for `fl_server.py`.

What it does:
- parses command-line arguments
- infers `input_dim` from the first CSV
- starts the server with:
  - number of rounds
  - number of clients
  - local epochs per round
  - address
  - save directory

Use this instead of calling `fl_server.py` manually.

### `fl/run_client.py`

This is the launcher for one Flower client.

What it does:
- maps `--id 1`, `--id 2`, `--id 3` to:
  - `client1.csv`
  - `client2.csv`
  - `client3.csv`
- starts a Flower client for that CSV

Typical usage:
- one terminal for `--id 1`
- one terminal for `--id 2`
- one terminal for `--id 3`

### `fl/run_fl_simulation.py`

This is the single-process simulation version of FL.

What it does:
- loads the same real HAR CSVs
- creates multiple simulated clients in one process
- uses the same model, criterion, local training, and aggregation logic
- writes `fl_history.json` after the simulation finishes

When to use it:
- use it if you want a fast local simulation
- it requires Flower simulation dependencies such as `ray`

### `fl/evaluate_global_model.py`

This evaluates the saved final global model after FL has completed.

What it does:
- loads `results/fl_rounds/global_model_latest.pth`
- evaluates it on each client CSV separately
- computes:
  - loss
  - accuracy
  - macro-F1
- writes:
  - `results/fl_rounds/global_model_eval.json`

Why this file matters:
- `fl_metrics.json` tells you round-by-round global performance
- `evaluate_global_model.py` tells you how the final checkpoint performs per client

## 3. Step-by-Step: Local Training Flow

This is what happens when you run:

```powershell
python fl/train_local.py --csv data/processed/client1.csv --epochs 20
```

### Step 1

`train_local.py` reads the CSV path and training arguments.

### Step 2

It calls `load_csv_data()` from `fl/data_utils.py`.

Inside that function:
- the CSV is loaded
- features and labels are separated
- labels are converted from `1..6` to `0..5`
- the data is split into local train/test subsets
- train features are standardized with `StandardScaler.fit_transform`
- test features are standardized with the same scaler using `transform`
- both splits are converted into PyTorch datasets and dataloaders

### Step 3

`train_local.py` builds a new model using:

```python
model = get_model(metadata.input_dim, metadata.num_classes)
```

So the local model starts from fresh random initialization.

### Step 4

It builds:
- loss function: `CrossEntropyLoss`
- optimizer: `Adam`

### Step 5

For each epoch:
- batches are read from the local train loader
- forward pass is computed
- loss is computed
- backpropagation runs
- optimizer updates local weights

### Step 6

After each epoch, the model is evaluated on the local test split using:
- accuracy
- macro-F1

### Step 7

If `--save` is provided:
- the local model is saved as `.pth`
- the epoch history is saved as JSON

## 4. Step-by-Step: How Server Weights Are Initialized

This is an important point.

In `fl/fl_server.py`, the strategy does **not** pass explicit `initial_parameters`.

That means Flower uses its default initialization flow.

What Flower does:
- checks whether the strategy already provided initial global weights
- if not, Flower requests initial parameters from one random available client

So in this project:
- the first global model is not manually created by the server
- it is pulled from one random client's freshly initialized model

Because all clients use the same architecture, this is valid.

Important implication:
- initial global weights are just random model weights from one client before any local training round begins

## 5. Step-by-Step: Federated Learning Round Flow

This is what happens when you run FL using:

```powershell
python fl/run_server.py
python fl/run_client.py --id 1
python fl/run_client.py --id 2
python fl/run_client.py --id 3
```

### Round 0: Setup

Server:
- starts Flower server
- creates `SaveModelStrategy`
- waits for clients

Clients:
- each client loads its own CSV
- each client creates its own local model, optimizer, and criterion

### Round 0.5: Initial Global Parameters

Because no explicit initial parameters are provided:
- Flower asks one random client for its current parameters
- those parameters become the initial global model

### Round 1: Server Sends Global Weights

The server sends the current global weights to each selected client.

On the client side:
- `set_parameters()` loads those weights into the local model

### Round 1: Local Client Training

Each client runs `fit()`.

Inside `fit()`:
- it reads `local_epochs` from the server config
- it trains for that many epochs on its own local train split
- it never sends raw data to the server

Only these are returned:
- updated weights
- number of examples
- metrics

### Round 1: Server Aggregation

The server receives weight updates from all participating clients.

Then FedAvg computes the new global model by averaging client updates using client sample counts.

After aggregation:
- `round_1.npz` is saved
- `global_model_round_1.pth` is saved
- `global_model_latest.pth` is updated

### Round 1: Evaluation

The server asks clients to evaluate the global model.

Each client:
- loads the new global model
- evaluates on its local test split
- returns:
  - test loss
  - accuracy
  - macro-F1
  - number of evaluation examples

### Round 1: Metrics Logging

The server aggregates those evaluation metrics using weighted averaging and writes them into:

```text
results/fl_rounds/fl_metrics.json
```

### Round 2 to Round N

The same cycle repeats:
- server sends current global weights
- clients train locally
- clients send updated weights
- server aggregates with FedAvg
- server saves the new global model
- clients evaluate
- server logs aggregated metrics

## 6. How FL Is Implemented in This Repo

The FL implementation is based on Flower's `NumPyClient` API plus `FedAvg`.

Client-side FL pieces:
- `get_parameters()`
- `set_parameters()`
- `fit()`
- `evaluate()`

Server-side FL pieces:
- `SaveModelStrategy(FedAvg)`
- `aggregate_fit()`
- `aggregate_evaluate()`
- `weighted_average()`

Core FL behavior:
- model stays local on each client during training
- only weights are exchanged
- server aggregates weights, not raw data
- evaluation is distributed across clients

## 7. Where Local Models Train

There are two different meanings of "local training" in this repo.

### A. Standalone local training

This happens in:
- `fl/train_local.py`

Purpose:
- test one client's CSV independently before FL

Model scope:
- one model per CSV
- no aggregation

### B. Federated local training

This happens in:
- `fl/fl_client.py`
- `fl/run_fl_simulation.py`

Purpose:
- each client trains locally during each FL round

Model scope:
- client receives current global model
- client trains locally for a few epochs
- client returns updated weights

## 8. What Gets Saved in `results/fl_rounds`

### `fl_metrics.json`

Per-round aggregated evaluation metrics for the global model.

Example meaning:
- round 10 entry = final global model metrics

### `fl_history.json`

Full Flower history from simulation mode.

Contains:
- distributed losses
- centralized losses
- distributed metrics
- centralized metrics

### `global_model_round_1.pth` ... `global_model_round_N.pth`

PyTorch checkpoints after each FL round.

### `global_model_latest.pth`

The latest global model checkpoint.

After the last round, this is the final global model.

### `round_1.npz` ... `round_N.npz`

Raw NumPy snapshots of the aggregated model weights.

### `global_model_eval.json`

Per-client evaluation of the final global model produced by `fl/evaluate_global_model.py`.

## 9. Recommended Working Sequence

### Step 1: Verify data exists

Make sure these exist:
- `data/processed/client1.csv`
- `data/processed/client2.csv`
- `data/processed/client3.csv`

### Step 2: Run standalone local training

Run:

```powershell
python fl/train_local.py --csv data/processed/client1.csv --epochs 20
python fl/train_local.py --csv data/processed/client2.csv --epochs 20
python fl/train_local.py --csv data/processed/client3.csv --epochs 20
```

Goal:
- confirm each client can train
- inspect accuracy and macro-F1

### Step 3: Run federated learning

Option A: real multi-process FL

Server terminal:

```powershell
python fl/run_server.py --rounds 10 --min-clients 3
```

Client terminals:

```powershell
python fl/run_client.py --id 1
python fl/run_client.py --id 2
python fl/run_client.py --id 3
```

Option B: simulation mode

```powershell
python fl/run_fl_simulation.py --rounds 10 --clients 3 --local-epochs 3
```

This requires Flower simulation dependencies.

### Step 4: Inspect round metrics

Read:
- `results/fl_rounds/fl_metrics.json`

### Step 5: Evaluate final global checkpoint

Run:

```powershell
python fl/evaluate_global_model.py
```

Then inspect:
- `results/fl_rounds/global_model_eval.json`

## 10. Current Final Model Result

From the current saved run:

- final global model checkpoint:
  - `results/fl_rounds/global_model_latest.pth`
- final round metrics:
  - loss `0.08803`
  - accuracy `0.96992`
  - macro-F1 `0.97037`

Per-client final evaluation:
- `client1.csv`
  - loss `0.12166`
  - accuracy `0.95813`
  - macro-F1 `0.95687`
- `client2.csv`
  - loss `0.08868`
  - accuracy `0.97230`
  - macro-F1 `0.97254`
- `client3.csv`
  - loss `0.05975`
  - accuracy `0.97745`
  - macro-F1 `0.97951`

## 11. Short Summary

If you want the shortest explanation:

- `data_utils.py` loads and prepares each client CSV
- `model.py` defines the HAR classifier
- `train_local.py` trains one client's model alone
- `fl_client.py` makes one client participate in FL
- `fl_server.py` aggregates client updates with FedAvg and saves checkpoints
- `run_server.py` starts the FL server
- `run_client.py` starts one FL client
- `run_fl_simulation.py` runs all clients in one process
- `evaluate_global_model.py` checks the final global model on all client CSVs

## 12. How To Add Differential Privacy

This section explains how differential privacy should be added to the current FL pipeline.

## 12.1 Where DP Fits In The Current Flow

Differential privacy should be applied on the client side, during local training, before model updates are sent back to the server.

That means:
- raw client data stays local because of federated learning
- client-side gradients or updates are privatized before they can influence the shared global model
- the server still performs FedAvg, but now it aggregates privacy-protected client updates

In this project, the best place to add DP is inside the local training path used by:
- `fl/train_local.py`
- `fl/fl_client.py`
- `fl/run_fl_simulation.py`

The server code in `fl/fl_server.py` usually does not need major DP logic changes if you are using client-side DP-SGD.

## 12.2 What DP Usually Means Here

For this codebase, the standard approach is:
- use DP-SGD on each client
- clip per-sample gradients
- add calibrated Gaussian noise
- track privacy budget such as epsilon and delta

This is commonly done using Opacus.

The basic idea is:
- normal local training:
  - compute gradients
  - update model
- DP local training:
  - compute per-sample gradients
  - clip them to a max norm
  - add random noise
  - then update model

So the model sent back after each FL round has been trained with privacy protection.

## 12.3 Which Files Need To Change

### `fl/train_local.py`

This is the first file to extend if you want local DP training.

What to add:
- command-line flags such as:
  - `--dp`
  - `--noise-multiplier`
  - `--max-grad-norm`
  - `--delta`
- logic to wrap the model, optimizer, and train loader with Opacus `PrivacyEngine`
- privacy budget reporting after each epoch

Why here:
- it lets you test DP locally on one client before enabling DP in FL

### `fl/fl_client.py`

This is the main FL file that needs DP.

What to add:
- initialize the privacy engine in the client constructor
- make the local optimizer private before round training begins
- optionally log:
  - epsilon
  - delta
  - noise multiplier
  - clipping norm

This is where client updates become privacy-protected during real federated learning.

### `fl/run_fl_simulation.py`

If you use simulation mode, this file should mirror the same DP behavior as `fl/fl_client.py`.

What to add:
- DP-enabled simulated clients
- same privacy config arguments
- optional privacy metrics in the output JSON

### `fl/fl_server.py`

This file may stay mostly unchanged for client-side DP.

Possible additions:
- include privacy settings in round config
- save aggregated privacy-related metadata
- write privacy metrics into `fl_metrics.json`

Important note:
- FedAvg itself does not provide differential privacy
- privacy comes from how each client performs local training

### `privacy/`

This repo already has a `privacy/` area in the project structure.

A clean design is to move DP-specific code there, for example:
- `privacy/dp_config.py`
- `privacy/dp_utils.py`
- `privacy/dp_train.py`

Then the FL code can call into those helpers instead of putting all DP logic directly into the FL files.

## 12.4 Recommended Implementation Design

The cleanest approach is:

### Step 1

Keep the current non-DP pipeline working as the baseline.

### Step 2

Add a reusable helper that attaches Opacus to a normal training setup.

For example, a helper could:
- accept `model`, `optimizer`, and `train_loader`
- return private versions of them
- expose a `privacy_engine` object

### Step 3

Use that helper first in `train_local.py`.

Goal:
- verify that local DP training works on one CSV
- confirm the model still trains
- measure the utility drop from DP

### Step 4

Reuse the same helper in `fl/fl_client.py` and `fl/run_fl_simulation.py`.

That keeps local standalone training and federated training aligned.

## 12.5 Exact Place In The Client Flow Where DP Is Applied

Current client flow:
- load CSV
- build model
- build optimizer
- train locally
- send updated weights

DP client flow:
- load CSV
- build model
- build optimizer
- attach privacy engine
- train locally with DP-SGD
- optionally compute epsilon
- send updated weights

So the difference is between:
- optimizer creation
- local epoch training

That is the point where gradient clipping and noise addition happen.

## 12.6 Conceptual Example Of The Code Change

Without DP:

```python
model = get_model(...)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(...)
```

With DP:

```python
model = get_model(...)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(...)

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm,
)
```

Then training continues mostly as before.

## 12.7 What Extra Results Should Be Saved

When DP is added, do not save only accuracy and loss.

Also save:
- epsilon
- delta
- noise multiplier
- max gradient norm
- whether secure RNG was used

Good output locations:
- local training history JSON
- `results/fl_rounds/fl_metrics.json`
- a separate DP report file such as:
  - `results/fl_rounds/fl_privacy.json`

## 12.8 What To Watch Out For

Adding DP changes training behavior.

Typical effects:
- training may become slower
- accuracy may decrease
- macro-F1 may decrease
- tuning becomes more sensitive

Common points to adjust:
- batch size
- learning rate
- number of local epochs
- clipping norm
- noise multiplier

Important caution:
- if you use DP-SGD, you should usually prefer optimizers and settings known to work well with Opacus
- if privacy accounting is important for your report, make sure epsilon is explicitly logged

## 12.9 Practical Integration Plan For This Repo

The most practical order for this repository is:

1. Add a reusable DP helper in `privacy/`
2. Enable DP in `fl/train_local.py`
3. Compare non-DP vs DP local training on each client CSV
4. Enable the same DP logic in `fl/fl_client.py`
5. Run federated training with DP
6. Save both utility metrics and privacy metrics
7. Evaluate the final global DP model using `fl/evaluate_global_model.py`

## 12.10 Short Summary For DP

If you want the shortest explanation:

- differential privacy should be added on the client side
- the best insertion point is local training, not server aggregation
- `train_local.py` should be updated first for testing
- `fl_client.py` and `run_fl_simulation.py` should then use the same DP training logic
- `fl_server.py` mainly stays as the FedAvg aggregator
- the final outputs should include both model quality metrics and privacy budget metrics
