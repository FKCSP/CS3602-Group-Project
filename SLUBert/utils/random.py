import random

import numpy as np
import torch


def set_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)