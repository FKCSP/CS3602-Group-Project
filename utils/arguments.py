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

def init_args(params=sys.argv[1:]) -> Arguments:
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    arg_parser.add_argument('--device', type=str, default='cuda',
                            help='Use which device: can be "cuda" (dafault) or "cpu"')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=200, help='terminate after maximum epochs')
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--seed', default=114514, type=int, help='Random seed')
    arg_parser.add_argument('--num_layer', default=2, type=int, help='number of layer')
    arg_parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='type of rnn')
    arg_parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    return arg_parser


arguments = init_args()
