import tensorflow as tf

class MNTDecoder(tf.contrib.seq2seq.Decoder):
    def __init__(self, encoder_input, attention_weights):