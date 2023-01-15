from typing import List, Tuple

import torch
from dataset.data import BIO, Label, LabelConverter


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
            if start != -1:
                value = ''.join(text[start:i])
                ret.append([act, slot, value])
                start = -1
            start = i
            act = v.act
            slot = v.slot
        elif v.bio == BIO.O and start != -1:
            value = ''.join(text[start:i])
            ret.append([act, slot, value])
            start = -1
        elif v.bio == BIO.I and (v.act, v.slot) != (act, slot):
            # invalid tag sequence
            return []
    return ret
