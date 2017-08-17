import tensorflow as tf

class MNTDecoder(tf.contrib.seq2seq.Decoder):
    def __init__(self, encoder_outputs, translation_vocab_size, translation_embedding_matrix, batch_size,
                 sos_id, eos_id, decoder_hidden_size=300, encoder_hidden_size=300, layers=2):
        self.encoder_outputs = encoder_outputs
        self.decoder_hidden_size = decoder_hidden_size
        self.translation_embedding_matrix = translation_embedding_matrix
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.batch_size = batch_size
        with tf.variable_scope("W1") as w1_scope:
            self.W1 = tf.get_variable("W1", shape=[2 * encoder_hidden_size, decoder_hidden_size], dtype=tf.float32)
            self.B1 = tf.get_variable("B1", shape=[2 * encoder_hidden_size], dtype=tf.float32)

        with tf.variable_scope("W2") as w2_scope:
            self.W2 = tf.get_variable("W2", shape=[decoder_hidden_size, 2 * encoder_hidden_size], dtype=tf.float32)
            self.B2 = tf.get_variable("B2", shape=[decoder_hidden_size], dtype=tf.float32)

        with tf.variable_scope("Out") as out_scope:
            self.out = tf.get_variable("Out", shape=[translation_vocab_size, 2 * decoder_hidden_size], dtype=tf.float32)
            self.out_bias = tf.get_variable("Out_Bias", shape=[translation_vocab_size])

        self.encoder_output_projected_cache = tf.matmul(self.W2, encoder_outputs, transpose_b=True)

        with tf.variable_scope("h_dec_lstm"):
            self.h_dec_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(decoder_hidden_size) for _ in range(layers)]
            )

    def initialize(self, name=None):
        with tf.variable_scope(name or "initialize"):
            finished_tensor = tf.zeros([self.batch_size], dtype=tf.bool)
            initial_input = tf.tile(
                tf.nn.embedding_lookup(self.translation_embedding_matrix, self.sos_id), [self.batch_size])
            initial_input = tf.concat([initial_input, tf.zeros(2 * self.decoder_hidden_size)], axis=1)
            initial_state = tf.zeros(self.decoder_hidden_size)

        return finished_tensor, initial_input, initial_state

    def step(self, time, inputs, state, name=None):
        with tf.variable_scope(name or "step"):
            decoder_output, _ = self.h_dec_lstm_cell(inputs, state)
            decoder_query = tf.nn.bias_add(tf.matmul(self.W1, decoder_output), self.B1)
            a = tf.nn.softmax(tf.matmul(self.encoder_outputs, decoder_query))
            decoder_output_context_adjusted = tf.concat([
                tf.nn.tanh(tf.nn.bias_add(tf.matmul(self.encoder_output_projected_cache, a), self.B2)),
                decoder_output
            ], axis=1)

            translation_vocab_logits = tf.nn.bias_add(tf.matmul(self.out, decoder_output_context_adjusted),
                                                      self.out_bias)
            translation_vocab_softmax = tf.nn.softmax(translation_vocab_logits)

            translation_id = tf.argmax(translation_vocab_softmax, axis=1)

            translation_embedding = tf.nn.embedding_lookup(self.encoder_output_projected_cache, translation_id)

            finished = translation_id == self.eos_id

        return (
            tf.contrib.seq2seq.BasicDecoderOutput(decoder_output_context_adjusted, translation_id),
            decoder_output,
            tf.concat([translation_embedding, decoder_output_context_adjusted], axis =1),
            finished
        )




