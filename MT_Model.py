import tensorflow as tf

class MT_Model(object):
    def __init__(self, translation_vocab_size, max_seq_len = 200, decoder_output_size=300 ,input_size=300, hidden_size=300, layers=2):
        self.translation_vocab_size = translation_vocab_size
        self.max_seq_len = max_seq_len
        self.decoder_output_size = decoder_output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers=layers
        self._add_placeholders()
        self._build_model()

    def _add_placeholders(self):
        self.glove_wordvectors = tf.placeholder(shape=[None, self.max_seq_len, self.input_size], dtype=tf.float32)
        self.labels_placeholder = tf.placeholder(shape=[None], dtype=tf.float32)

    def _build_model(self):
        self.encoded_sequence = self._run_lstm_encoder()
        self.decoder_sequence = self._run_decoder()

    def _run_lstm_encoder(self):
        with tf.variable_scope("mt_lstm"):
            with tf.variable_scope("forwards"):
                forward_lstm_cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size) for _ in range(self.layers)])
            with tf.variable_scope("backwards"):
                backward_lstm_cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size) for _ in range(self.layers)])

            fw_encoding, bw_encoding, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_lstm_cells,
                cell_bw=backward_lstm_cells,
                inputs=self.glove_wordvectors
            )

        return tf.concat([fw_encoding, bw_encoding], axis=2)

    def _run_decoder(self):
        with tf.variable_scope("h_dec_lstm"):
            h_dec_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size*2) for _ in range(self.layers)])

        with tf.variable_scope("W1") as w1_scope:
            W1 = tf.get_variable("W1", shape=[2*self.hidden_size, self.hidden_size], dtype=tf.float32)
            B1 = tf.get_variable("B1", shape=[2*self.hidden_size], dtype=tf.float32)

        with tf.variable_scope("W2") as w2_scope:
            W2 = tf.get_variable("W2", shape=[self.hidden_size, 2*self.hidden_size], dtype=tf.float32)
            B2 = tf.get_variable("B2", shape=[self.hidden_size], dtype=tf.float32)

        with tf.variable_scope("Out") as out_scope:
            out = tf.get_variable("Out", shape=[self.translation_vocab_size, 2 * self.hidden_size], dtype=tf.float32)

        pre_computed_w2_h = tf.matmul(W2, self.encoded_sequence, transpose_b=True)

