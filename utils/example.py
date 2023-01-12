import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        dataset = json.load(open(data_path, 'r'))
        examples = []
        for data in dataset:
            conv = conversation()
            for utt in data:
                conv.app(cls(utt))
            examples.append(conv)
        return examples

    def __init__(self, ex: dict):
        self.utt = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]


class conversation():
    def __init__(self):
        self.utt = ""
        self.input_idx = []
        self.tag_id = []
        self.ex_lst = []

    def app(self, ex):
        if self.utt:
            self.utt += ';'
            self.input_idx.append(2)
            self.tag_id.append(1)
        self.utt += ex.utt
        self.input_idx += ex.input_idx
        self.tag_id += ex.tag_id
        self.ex_lst.append(ex)
