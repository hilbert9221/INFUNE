import pickle
from math import ceil
import random


def write_pickle(obj, outfile: str, protocol: int = -1) -> None:
    """
    A wrapper for pickle.dump().
    """
    with open(outfile, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def read_pickle(infile: str):
    """
    A wrapper for pickle.load().
    """
    with open(infile, 'rb') as f:
        return pickle.load(f)


def split_list(input: list, batch_size: int, shuffle_input: bool = True) -> list:
    """
    A anology to torch.utils.data.DataLoader that takes a list as the input.

    Args:
        input: a list of samples
        batch_size: batch size
        shuffle_input: shuffle or not

    Return:
        a list of batches, each batch is of length batch size
    """
    num = ceil(len(input) / batch_size)
    if shuffle_input:
        random.shuffle(input)
    return [input[i * batch_size: (i + 1) * batch_size] for i in range(num)]
