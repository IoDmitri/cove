import numpy as np
import tensorflow as tf

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


def main(argv=None):
    if FLAGS.mode == "debug":
        parallel_dict = {
            "a": "z",
            "b": "y",
            "c": "x",
            "d": "w"
        }

        choices = list(parallel_dict.keys())

        initial_sequences = [[choices[np.random.choice(len(choices), 1)[0]] for _ in range(10)] for _ in range(10)]
        initial_targets = [[parallel_dict[letter] for letter in seq] for seq in initial_sequences]



