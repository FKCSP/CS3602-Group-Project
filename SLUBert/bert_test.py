import os
import sys

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from SLUBert.model.decoder import SimpleDecoder
from SLUBert.utils.arguments import arguments
from SLUBert.dataset.data import BIO, Label, LabelConverter, MyDataLoader, MyDataset

from datetime import datetime
from SLUBert.utils.logger import Logger
from SLUBert.utils.initialization import args_print

def get_output(text: List[str], output: torch.Tensor, label_converter: LabelConverter) -> List[Tuple[str, str, str]]:
    ret = []
    output = output[1:-1].argmax(dim=1)
    labels = [label_converter.index_to_label(i.item()) for i in output]
    labels.append(Label(BIO.O, '', ''))
    start = -1
    act = ''
    slot = ''
    for i, v in enumerate(labels):
        if v.bio == BIO.B:
            start = i
            act = v.act
            slot = v.slot
        elif v.bio == BIO.O and start != -1:
            value = ''.join(text[start:i])
            ret.append([act, slot, value])
        elif v.bio == BIO.I and (v.act, v.slot) != (act, slot):
            # invalid tag sequence
            return []
    return ret


def set_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

# set random seed
set_random_seed(arguments.seed)

# label converter
label_converter = LabelConverter('data/ontology.json')
pretrained_model_name = 'bert-base-chinese'

# make directory
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)

# prepare dataset & dataloader
train_dataset = MyDataset('data/train.json', label_converter, pretrained_model_name, cache_dir)
dev_dataset = MyDataset('data/development.json', label_converter, pretrained_model_name, cache_dir)
train_data_loader = MyDataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True)
dev_data_loader = MyDataLoader(dev_dataset)
encoding_len = train_dataset[0][0][0].vector_with_noise.shape[1]

# model configuration
decoder = SimpleDecoder(encoding_len, label_converter.num_indexes, arguments).to(arguments.device)
optimizer = Adam(decoder.parameters(), lr=arguments.lr, weight_decay=arguments.weight_decay)
loss_fn = nn.CrossEntropyLoss()

# logger information
datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name = f'bert.lr_{arguments.lr}.rnn_{arguments.rnn}.hidden_{arguments.hidden_size}.layer_{arguments.num_layer}.batch_{arguments.batch_size}.seed_{arguments.seed}.{datetime_now}'
exp_dir = os.path.join('result/', experiment_name)
os.makedirs(exp_dir, exist_ok=True)
logger = Logger.init_logger(filename=exp_dir + '/train.log')
args_print(arguments, logger)


for epoch in range(arguments.max_epoch):
    logger.info(f'Epoch: {epoch}')
    total_loss = 0
    # training
    decoder.train()
    for batch_x, batch_y in train_data_loader:
        batch_loss = 0
        for round_x, round_y in zip(batch_x, batch_y):
            for x, y in zip(round_x, round_y):
                output = decoder(x.vector_without_noise)
                loss = loss_fn(output, y)
                total_loss += loss
                batch_loss += loss
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    avgloss = total_loss.item() / len(train_dataset)
    logger.info(f'train. loss: {avgloss}')

    # validation
    n_total = 0
    n_correct = 0
    decoder.eval()
    with torch.no_grad():
        for batch_x, batch_y in dev_data_loader:
            for round_x, round_y in zip(batch_x, batch_y):
                for x, y in zip(round_x, round_y):
                    n_total += 1
                    output = decoder(x.vector_with_noise)
                    prediction = get_output(x.tokens_with_noise, output, label_converter)
                    expected = get_output(x.tokens_without_noise, y, label_converter)
                    if prediction == expected:
                        n_correct += 1
    acc = 100*n_correct / n_total
    logger.info(f'Acc: {acc} correct: {n_correct} total: {n_total}')
