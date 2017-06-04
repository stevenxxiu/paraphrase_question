#!/bin/bash
#PBS -q gpupascal
#PBS -l ngpus=1
#PBS -l ncpus=6
#PBS -l mem=16GB
#PBS -l walltime=20:00:00
module load tensorflow/1.0.1-python3.5
cd /short/cp1/sx6361/projects/paraphrase_question
python=/home/563/sx6361/.pyenv/versions/3.6.1/bin/python
export PYTHONPATH="/home/563/sx6361/.pyenv/versions/3.6.1/lib/python3.6/site-packages:$PWD"
export PYTHONUNBUFFERED=1
"$python" paraphrase_question/main.py decatt '{
    "intra_sent": false, "emb_size": 300, "emb_glove": false, "context_size": 1,
    "n_intra": [400, 200], "n_intra_bias": 10, "n_attend": [400, 200], "n_compare": [400, 200], "n_classif": [400, 200],
    "dropout_rate": 0.1, "lr": 0.1, "batch_size": 64, "epoch_size": 100
}'
