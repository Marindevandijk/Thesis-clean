#!/bin/bash

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

python -m venv thesis_env
source thesis_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt