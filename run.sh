#!/bin/bash

# Baseline
python -m scripts.slu_baseline --encoder_cell LSTM --dropout 0.1 --runs 5 --device 0
python -m scripts.slu_baseline --encoder_cell GRU --dropout 0.1 --runs 5 --device 0
python -m scripts.slu_baseline --encoder_cell RNN --dropout 0 --runs 5 --device 0