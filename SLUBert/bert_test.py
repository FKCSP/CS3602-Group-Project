import os
from datetime import datetime
from typing import List, Tuple

import torch
from dataset.data import BIO, Label, LabelConverter, MyDataLoader, MyDataset
from model.decoder import SimpleDecoder
from torch import nn
from torch.optim import Adam
from utils.arguments import arguments
from utils.evaluator import Evaluator
from utils.initialization import args_print
from utils.logger import Logger
from utils.output import get_output
from utils.random import set_random_seed

# set random seed
set_random_seed(arguments.seed)

# prepare dataset & dataloader
label_converter = LabelConverter('data/ontology.json')
pretrained_model_name = 'bert-base-chinese'
cache_dir = 'cache'
train_dataset = MyDataset('data/train.json', label_converter, pretrained_model_name, cache_dir)
dev_dataset = MyDataset('data/development.json', label_converter, pretrained_model_name, cache_dir)
train_data_loader = MyDataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True)
dev_data_loader = MyDataLoader(dev_dataset)
encoding_len = train_dataset[0][0][0].vector_with_noise.shape[1]

# model configuration
decoder = SimpleDecoder(encoding_len, label_converter.num_indexes).to(arguments.device)
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
    avg_loss = total_loss.item() / len(train_dataset)
    logger.info(f'train. loss: {avg_loss}')

    # validation
    evaluator = Evaluator()
    decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in dev_data_loader:
            for round_x, round_y in zip(batch_x, batch_y):
                for x, y in zip(round_x, round_y):
                    output = decoder(x.vector_without_noise)
                    loss = loss_fn(output, y)
                    total_loss += loss
                    prediction = get_output(x.tokens_with_noise, output, label_converter)
                    expected = get_output(x.tokens_without_noise, y, label_converter)
                    evaluator.add_result(prediction, expected)
    acc = evaluator.accuracy_rate
    f1_score = evaluator.f1_score
    avg_loss = total_loss.item() / len(dev_dataset)
    logger.info(f'Acc: {acc:.2f}, F1 Score: {f1_score:.2f}, Avg. Loss: {avg_loss:.5f}')
