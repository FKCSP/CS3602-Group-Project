'''
Usage: from <this> import arguments
'''

import argparse
import sys
from typing import NoReturn


class Arguments:
    '''Just for type hints. Don't initialize it.'''

    def __init__(self) -> NoReturn:
        raise NotImplementedError()

    device: str
    lr: float
    max_epoch: int
    batch_size: int
    seed: int
    num_layer: int
    noise: bool


def init_args(params=sys.argv[1:]) -> Arguments:
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--device', type=str, default='cuda',
                            help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--noise', type=int, default='1')
    #### Training Hyperparams ####
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    arg_parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    arg_parser.add_argument('--max_epoch', type=int, default=150, help='terminate after maximum epochs')
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--num_layer', default=1, type=int, help='number of layer')
    arg_parser.add_argument('--rnn', default='GRU', choices=['LSTM', 'GRU', 'RNN'], help='type of rnn')
    arg_parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    return arg_parser


arguments = init_args()
