from typing import Sequence


class Evaluator:
    def __init__(self):
        self._n_sentences = 0
        self._n_correct_sentences = 0
        self._n_prediction_tags = 0
        self._n_truth_tags = 0
        self._n_correct_tags = 0

    def add_result(self, pred: Sequence, truth: Sequence) -> None:
        pred = set([tuple(i) for i in pred])
        truth = set([tuple(i) for i in truth])
        self._n_sentences += 1
        if pred == truth:
            self._n_correct_sentences += 1
        self._n_prediction_tags += len(pred)
        self._n_truth_tags += len(truth)
        self._n_correct_tags += len(pred & truth)

    @property
    def precision_rate(self) -> float:
        if self._n_prediction_tags == 0:
            return 0.0
        return self._n_correct_tags / self._n_prediction_tags

    @property
    def recall_rate(self) -> float:
        if self._n_truth_tags == 0:
            return 0.0
        return self._n_correct_tags / self._n_truth_tags

    @property
    def accuracy_rate(self) -> float:
        return self._n_correct_sentences / self._n_sentences

    @property
    def f1_score(self) -> float:
        p = self.precision_rate
        r = self.recall_rate
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
