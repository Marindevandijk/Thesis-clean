# Inferring Equations of Motion of Fighting Zebrafish

This repository contains the analysis code for my thesis project on inferring effective equations of motion from interacting zebrafish trajectories. The project focuses on fighting zebrafish and uses the relative distance and relative orientation of two fish to infer force fields with Stochastic Force Inference.

The main state variables are:

* `D`: distance between the fish, measured between pectoral points
* `theta1`: relative orientation angle of fish 1 with respect to fish 2
* `theta2`: relative orientation angle of fish 2 with respect to fish 1

The inferred force field describes the deterministic part of the dynamics in this three-dimensional state space.

## Repository structure

```text
Thesis-clean/
│
├── README.md
├── requirements.txt
├── setup_env.sh
├── .gitignore
│
├── SFI/
│   └── Stochastic Force Inference code
│
├── scripts/
│   ├── inference/
│   │   └── Python scripts for running inference and simulations
│   └── slurm/
│       └── SLURM scripts for running jobs on a cluster
│
├── notebooks/
│   └── Exploratory notebooks
│
├── data/
│   └── Local input data, not included in the repository
│
└── results/
    └── Local output files, not included in the repository
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Marindevandijk/Thesis-clean.git
cd Thesis-clean
```

Create and activate a virtual environment:

```bash
python -m venv thesis_env
source thesis_env/bin/activate
```

Install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The required Python packages are listed in `requirements.txt`.

## Data

The raw tracking data are not included in this repository because the files are large. The scripts expect the tracking files to be stored locally in:

```text
Data/tracking_results/
```

Expected input files include tracking results such as:

```text
FishTank20200130_153857_tracking_results.h5
FishTank20200130_181614_tracking_results.h5
FishTank20200213_154940_tracking_results.h5
...
```

The scripts also use metadata files such as:

```text
fightBouts.h5
winners_losers_inconclusive.h5
```

Before running the analysis, make sure the local folder structure matches the paths used in the scripts.

## Stochastic Force Inference dependency

This project uses Stochastic Force Inference (SFI) to infer the deterministic force field from stochastic trajectory data. The SFI method is based on Frishman and Ronceray (2020) [1]. The SFI code used for this thesis is included in this repository in:

```text
SFI/
```

## Running the analysis

Example: run an inference script locally from the root of the repository:

```bash
python scripts/inference/Run_all_ordered.py
```

For cluster use, submit the corresponding SLURM script:

```bash
sbatch scripts/slurm/run_all_ordered.slurm
```

The scripts perform the following steps:

1. Load tracked 3D zebrafish coordinates from `.h5` files from the fighting-zebrafish dataset [2].
2. Select fight-bout data.
3. Compute the state variables `D`, `theta1`, and `theta2`.
4. Clean and segment the trajectory data.
5. Infer the force field using Stochastic Force Inference [1].
6. Simulate stochastic trajectories from the inferred model.
7. Save model outputs, endpoint data, figures, and metadata.

## Output

The output is written to local results folders, for example:

```text
Results/
Results_Spars/
Results_newbasis/
```

These folders are ignored by git and should not be uploaded to GitHub. Typical output files include:

```text
metadata.txt
stochastic_simulated_trajectories.npz
SFI_model_data_*.npz
Endpoints_*.csv
stochastic_simulation_distributions.png
```

## Notes on reproducibility
 Some scripts are specific to particular experiments, fight bouts, or basis choices. The file names indicate the analysis condition, such as:

* `Run_all_ordered.py`: analysis using ordered winner-loser coordinates and on whole fightbout
* `Run_25%_...py`: analysis on a subset of the data
* `Run_exp*.py`: analysis for individual experiments

## Important git information

Large files should not be committed to the repository. This includes:

```text
Data/
Results/
*.npz
*.csv
*.log
thesis_env/
__pycache__/
```

These files are excluded in `.gitignore`.

## References

[1] Anna Frishman and Pierre Ronceray. Learning force fields from stochastic trajectories. Physical Review X, 10(2):021009, 2020.

[2] Liam O’Shaughnessy, Tatsuo Izawa, Ichiro Masai, Joshua W. Shaevitz, and Greg J. Stephens. Dynamics of dominance in interacting zebrafish. PRX Life, 2(4):043006, 2024.

## Author

Marinde van Dijk

Thesis project, 2026.

