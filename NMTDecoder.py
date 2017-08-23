import tensorflow as tf

class NMTDecoder(tf.contrib.seq2seq.Decoder):
    def __init__(self, encoder_outputs, translation_vocab_size, translation_embedding_matrix, batch_size,
                 sos_id, eos_id, decoder_hidden_size=300, encoder_hidden_size=300, layers=2):
        self.encoder_outputs = encoder_outputs
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.translation_vocab_size = translation_vocab_size
        self.translation_embedding_matrix = translation_embedding_matrix
        self.sos_id = sos_id
        self.eos_id = eos_id
        self._batch_size = batch_size
        with tf.variable_scope("W1") as w1_scope:
            self.W1 = tf.get_variable("W1", shape=[decoder_hidden_size, 2 * encoder_hidden_size], dtype=tf.float32)
            self.B1 = tf.get_variable("B1", shape=[2 * encoder_hidden_size], dtype=tf.float32)

        with tf.variable_scope("W2") as w2_scope:
            self.W2 = tf.get_variable("W2", shape=[decoder_hidden_size, 2 * encoder_hidden_size], dtype=tf.float32)
            self.B2 = tf.get_variable("B2", shape=[decoder_hidden_size], dtype=tf.float32)

        with tf.variable_scope("Out") as out_scope:
            self.out = tf.get_variable("Out", shape=[2 * decoder_hidden_size, translation_vocab_size], dtype=tf.float32)
            self.out_bias = tf.get_variable("Out_Bias", shape=[translation_vocab_size])

        encoder_outputs = tf.transpose(encoder_outputs, [0,2,1])
        self.encoder_output_projected_cache = tf.einsum("ij,ajk->aik", self.W2, encoder_outputs)

        with tf.variable_scope("h_dec_lstm"):
            self.h_dec_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(decoder_hidden_size) for _ in range(layers)],
                state_is_tuple=True
            )

    def initialize(self, name=None):
        with tf.variable_scope(name or "initialize"):
            finished_tensor = tf.zeros([self.batch_size], dtype=tf.bool)
            sos_embedding = tf.reshape(tf.nn.embedding_lookup(self.translation_embedding_matrix, self.sos_id), [1, -1])
            initial_input = tf.tile(sos_embedding, [self.batch_size, 1])
            initial_input = tf.concat([initial_input, tf.zeros([self.batch_size, 2*self.decoder_hidden_size])], axis=1)

            # print(f"finished_tensor - {finished_tensor}")
            # print(f"initial_input - {initial_input}")
            # print(f"zerp_state - {self.h_dec_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)}")
        return finished_tensor, initial_input, self.h_dec_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

    def step(self, time, inputs, state, name=None):
        # print(f"inputs - {inputs}")
        with tf.variable_scope(name or "step"):
            decoder_output, lstm_state = self.h_dec_lstm_cell(inputs, state)
            # print(f"decoder_output - {decoder_output}")
            decoder_query = tf.nn.bias_add(tf.matmul(decoder_output, self.W1), self.B1, name="decoder_query")
            # print(f"lstm_state - {lstm_state}")
            # print(f"decoder_query - {decoder_query}")
            decoder_query = tf.expand_dims(decoder_query, axis=2)
            a = tf.nn.softmax(tf.matmul(self.encoder_outputs, decoder_query, name="a"))
            # print(f"a - {a}")
            # print(f"encoder_output_projected_cache - {self.encoder_output_projected_cache}")
            # print(f"matmul - {tf.matmul(self.encoder_output_projected_cache, a)}")
            decoder_output_context_adjusted = tf.concat([
                tf.nn.tanh(tf.nn.bias_add(tf.squeeze(tf.matmul(self.encoder_output_projected_cache, a, name="inner_tanh")),
                                          self.B2)),
                decoder_output
            ], axis=1, name="decoder_output_context_concat")

            translation_vocab_logits = tf.nn.bias_add(tf.matmul(decoder_output_context_adjusted, self.out),
                                                      self.out_bias)
            translation_vocab_softmax = tf.nn.softmax(translation_vocab_logits)

            translation_id = tf.argmax(translation_vocab_softmax, axis=1)

            translation_embedding = tf.nn.embedding_lookup(self.translation_embedding_matrix, translation_id)

            finished = translation_id == self.eos_id
            # finished = tf.Print(finished, [finished], "finished tensor")
            # print(f"basicDeocderOutput - {tf.contrib.seq2seq.BasicDecoderOutput(translation_vocab_logits, translation_id)}")
            # print(f"decoder_output - {decoder_output}")
            # print(f"context_adjusted - {tf.concat([translation_embedding, decoder_output_context_adjusted], axis=1)}")
            # print(f"finished - {finished}")
        return (
            translation_vocab_logits,
            lstm_state,
            tf.concat([translation_embedding, decoder_output_context_adjusted], axis=1, name="return_concat"),
            finished
        )

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def output_size(self):
        #return self.decoder_hidden_size
        return self.translation_vocab_size
    @property
    def output_dtype(self):
        return tf.float32




