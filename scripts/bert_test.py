import os
import sys

from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

import torch
from torch import nn

from model.fnn_decoder import FNNDecoder
from utils.arguments import arguments
from utils.data import LabelConverter, MyDataLoader, MyDataset

label_converter = LabelConverter('data/ontology.json')
train_dataset = MyDataset('data/train.json', label_converter, 'bert-base-chinese', asr_output=False)
dev_dataset = MyDataset('data/development.json', label_converter, 'bert-base-chinese', asr_output=False)
train_data_loader = MyDataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True)
dev_data_loader = MyDataLoader(dev_dataset)
encoding_len = train_dataset[0][0][0].shape[1]
decoder = FNNDecoder(encoding_len, label_converter.num_indexes).to(arguments.device)
optimizer = Adam(decoder.parameters(), arguments.lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(arguments.max_epoch):
    print('epoch:', epoch)
    for batch_x, batch_y in train_data_loader:
        optimizer.zero_grad()
        for round_x, round_y in zip(batch_x, batch_y):
            for x, y in zip(round_x, round_y):
                output = decoder(x)
                loss = loss_fn(output, y)
                loss.backward()
        optimizer.step()

    # test
    n_total = 0
    n_correct = 0
    for batch_x, batch_y in dev_data_loader:
        with torch.no_grad():
            for round_x, round_y in zip(batch_x, batch_y):
                for x, y in zip(round_x, round_y):
                    n_total += 1
                    output = decoder(x)[1:-1]
                    prediction = output.argmax(dim=1)
                    expected = y[1:-1].argmax(dim=1)
                    a = torch.Tensor()
                    if prediction.equal(expected):
                        n_correct += 1
    print(n_correct, n_total, n_correct / n_total)
