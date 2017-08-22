import sys

import numpy as np
import tensorflow as tf

from data_utils import *
from MT_Model import MT_Model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("mode", "debug",
                           """
                           The mode in which to run. Available modes are "train", and "debug" [train]
                           """)
tf.app.flags.DEFINE_integer("enc_h", 300,
                            """
                            The hidden size of the encoder [300]
                            """)
tf.app.flags.DEFINE_integer("dec_h", 300,
                            """
                            the hidden size of the decoder [300]
                            """)
tf.app.flags.DEFINE_integer("batch_size", 128,
                            """
                            The size of the batch to use for training [128]
                            """)
tf.app.flags.DEFINE_integer("epochs", 20,
                            """
                            The max number of epochs to run [20]
                            """)

lr = 1.0
def _run_epoch(model, data, sess, batch_size, train_op=None, verbose=10):
    if train_op is None:
        train_op = tf.no_op()

    train_loss = []
    total_steps = sum(1 for _ in data_iterator(data, batch_size))
    for step, (initial_sequence, labels) in enumerate(data_iterator(data, batch_size)):
        feed = {
            model.encoded_sequence_placeholder: initial_sequence,
            model.labels_placeholder: labels,
            model.learning_rate_placeholder: lr
        }

        loss, _ = sess.run([model.loss_op, train_op], feed_dict=feed)
        train_loss.append(loss)

        if verbose and step % verbose == 0:
            sys.stdout.write("\r{} / {}: loss = {}".format(step, total_steps, np.mean(train_loss)))
            sys.stdout.flush()

    return np.mean(train_loss)

def main(argv=None):

    if FLAGS.mode == "debug":
        parallel_dict = {
            "a": "z",
            "b": "y",
            "c": "x",
            "d": "w"
        }

        choices = list(parallel_dict.keys())

        initial_sequences = [[choices[np.random.choice(len(choices), 1)[0]] for _ in range(10)] for _ in range(500)]
        initial_targets = [[parallel_dict[letter] for letter in seq] for seq in initial_sequences]

        translation_vocab_size = len(parallel_dict) + 2

        initial_seq_to_number = {"a": 1, "b": 2, "c" : 3, "d" : 4}
        translation_seq_to_number = {"z": 3, "y": 4, "x": 5, "w": 6}

        initial_sequences = [[initial_seq_to_number[token] for token in seq] for seq in initial_sequences]
        initial_targets = [[translation_seq_to_number[token] for token in seq] for seq in initial_targets]

        random_embedding_matrix = np.random.rand(len(choices) + 1, FLAGS.enc_h)
        initial_sequences = np.array(initial_sequences)

        initial_sequences = random_embedding_matrix[initial_sequences, :]
        
        data = (initial_sequences, initial_targets)


    model = MT_Model(translation_vocab_size=translation_vocab_size, max_seq_encoder_len=10, max_seq_decoder_len=10,
                     sos_id=1, eos_id=2)

    with tf.Session as sess:
        start = tf.global_variables_initializer()
        sess.run(start)

        for epoch in FLAGS.epochs:
            train_loss = _run_epoch(model, data, sess, FLAGS.batch_size, model.train_op)
            print(f"epoch - {epoch}")
            print(f"train_loss for epoch - {train_loss}")


if __name__ == "__main__":
	main(sys.argv[1:])
