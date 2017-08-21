import numpy as np
import tensorflow as tf

def data_iterator(data, batch_size):
    seq = data[0]
    labels = data[1]
    batches = len(seq) // batch_size
    remainder = len(seq) % batch_size
    if remainder != 0:
        batches += 1

    for seq, labels in zip(np.array_split(seq, batches), np.array_split(labels, batches)):
        if len(seq) < batch_size:
            batch_deficiency = batch_size - len(seq)

            seq = np.pad(seq, [(0, batch_deficiency), (0,0), (0,0)], "constant")
            labels = np.pad(labels, [(0, batch_deficiency), (0,0), (0,0)], "constant")

        yield seq, labels
