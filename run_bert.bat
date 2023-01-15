@echo off
set PYTHONPATH=%cd%\SLUBert
python SLUBert\bert_test.py --rnn=GRU
python SLUBert\bert_test.py --rnn=LSTM
python SLUBert\bert_test.py --rnn=RNN
python SLUBert\bert_multi_turn_test.py --rnn=GRU