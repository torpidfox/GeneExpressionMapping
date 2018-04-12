import numpy as np
from random import shuffle

def _pad(arr, batch_size, padding_val = 20):
	result = np.append(arr, arr[:padding_val-batch_size, :], axis=0)
	return result

def _parse_text_file(batch_size, file, batch_count):
    with open(file, 'r') as f:
        result = [_pad(np.asarray([[np.float64(el) for el in f.readline().split()] for _ in range(batch_size)]), batch_size) for _ in range(batch_count)]

    shuffle(result)
    return result

def construct_dataset(batch_size, filenames, batch_count=1):
    for file in filenames:
        yield _parse_text_file(batch_size, file, batch_count)