# utils.py

import numpy as np
import random
import tensorflow as tf

# set seed in different cases to ensure randomness
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)