import os
from datetime import datetime

import torch
from dataset.data import LabelConverter, MyDataLoader, MyDataset
from model.decoder import SimpleDecoder
from torch import nn
from torch.optim import Adam
from utils.arguments import arguments
from utils.evaluator import Evaluator
from utils.initialization import args_print
from utils.logger import Logger
from utils.output import get_output
from utils.random import set_random_seed

os.makedirs('trained-models', exist_ok=True)

random_seeds = [99, 999, 9999, 99999, 114514]

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
experiment_name = f'bert.lr_{arguments.lr}.rnn_{arguments.rnn}.hidden_{arguments.hidden_size}.layer_{arguments.num_layer}.{datetime_now}'
exp_dir = os.path.join('result/', experiment_name)
os.makedirs(exp_dir, exist_ok=True)
logger = Logger.init_logger(filename=exp_dir + '/train.log')
args_print(arguments, logger)


best_accuracies = []
best_f1_scores = []
best_precisions = []
best_recalls = []

for run, seed in enumerate(random_seeds):
    set_random_seed(seed)
    best_acc = 0
    for epoch in range(arguments.max_epoch):
        # logger.info(f'Epoch: {epoch}')
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
        # logger.info(f'train. loss: {avg_loss}')

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
                        input_tokens = x.tokens_with_noise if arguments.noise else x.tokens_without_noise
                        prediction = get_output(input_tokens, output, label_converter)
                        expected = get_output(x.tokens_without_noise, y, label_converter)
                        evaluator.add_result(prediction, expected)
        acc = evaluator.accuracy_rate
        f1_score = evaluator.f1_score
        avg_loss = total_loss.item() / len(dev_dataset)
        # logger.info(f'Acc: {acc:.5f}, F1 Score: {f1_score:.5f}, Avg. Loss: {avg_loss:.5f}')
        if acc > best_acc:
            # logger.info('New best!')
            best_acc = acc
            best_f1_score = f1_score
            best_precision = evaluator.precision_rate
            best_recall = evaluator.recall_rate
            torch.save({
                'epoch': epoch,
                'model': decoder.state_dict(),
                'optim': optimizer.state_dict(),
                'seed': seed,
                'run': run,
            }, 'trained-models/slu-bert.bin')
    best_accuracies.append(best_acc)
    best_f1_scores.append(best_f1_score)
    best_precisions.append(best_precision)
    best_recalls.append(best_recall)

logger.info("Dev ACC:{:.2f}-+-{:.2f} Precision:{:.2f}-+-{:.2f} Recall:{:.2f}-+-{:.2f} F score:{:.2f}-+-{:.2f}".format(
    torch.tensor(best_accuracies).mean(), torch.tensor(best_accuracies).std(),
    torch.tensor(best_precisions).mean(), torch.tensor(best_precisions).std(),
    torch.tensor(best_recalls).mean(), torch.tensor(best_recalls).std(),
    torch.tensor(best_f1_scores).mean(), torch.tensor(best_f1_scores).std(),
))
