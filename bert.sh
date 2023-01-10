#!/bin/bash

python -m scripts.bert_test --rnn LSTM --num_layer 2 --lr 0.1
python -m scripts.bert_test --rnn LSTM --num_layer 2 --lr 0.01
python -m scripts.bert_test --rnn LSTM --num_layer 2 --lr 0.001

python -m scripts.bert_test --rnn GRU --num_layer 2 --lr 0.1
python -m scripts.bert_test --rnn GRU --num_layer 2 --lr 0.01
python -m scripts.bert_test --rnn GRU --num_layer 2 --lr 0.001

python -m scripts.bert_test --rnn RNN --num_layer 2 --lr 0.1
python -m scripts.bert_test --rnn RNN --num_layer 2 --lr 0.01
python -m scripts.bert_test --rnn RNN --num_layer 2 --lr 0.001

python -m scripts.bert_test --rnn LSTM --num_layer 1 --lr 0.1
python -m scripts.bert_test --rnn LSTM --num_layer 1 --lr 0.01
python -m scripts.bert_test --rnn LSTM --num_layer 1 --lr 0.001

python -m scripts.bert_test --rnn GRU --num_layer 1 --lr 0.1
python -m scripts.bert_test --rnn GRU --num_layer 1 --lr 0.01
python -m scripts.bert_test --rnn GRU --num_layer 1 --lr 0.001

python -m scripts.bert_test --rnn RNN --num_layer 1 --lr 0.1
python -m scripts.bert_test --rnn RNN --num_layer 1 --lr 0.01
python -m scripts.bert_test --rnn RNN --num_layer 1 --lr 0.001