import tensorflow as tf
from NMTDecoder import NMTDecoder

class MT_Model(object):
    def __init__(self, translation_vocab_size, batch_size=128, max_seq_encoder_len = 200, max_seq_decoder_len=200,
                 decoder_output_size=300, input_size=300, hidden_size=300, layers=2, sos_id=1, eos_id=2):
        self.translation_vocab_size = translation_vocab_size
        self.batch_size = batch_size
        self.max_seq_encoder_len = max_seq_encoder_len
        self.max_decoder_seq_len =  max_seq_decoder_len
        self.decoder_output_size = decoder_output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers=layers
        self.sos_id = sos_id
        self.eos_id = eos_id
        self._add_translation_embedding_matrix()
        self._add_placeholders()
        self._build_model()

    def _add_translation_embedding_matrix(self):
        with tf.variable_scope("translation") as scope:
            self.translation_embedding_matrix = tf.get_variable("Translation_embedding", shape=[self.translation_vocab_size, self.input_size])

    def _add_placeholders(self):
        self.encoded_sequence_placeholder = tf.placeholder(shape=[None, self.max_seq_encoder_len, self.input_size], dtype=tf.float32)
        self.labels_placeholder = tf.placeholder(shape=[None, self.max_decoder_seq_len], dtype=tf.float32)
        self.learning_rate_placeholder = tf.placeholder(shape=(), dtype=tf.float32)

    def _build_model(self):
        self.encoded_sequence = self._run_lstm_encoder()
        self.translated_sequence, self.translated_seq_lengths = self._run_decoder()
        self.loss_op = self._add_loss(self.translated_sequence)
        self.train_op = self._add_train_step(self.loss_op)

    def _run_lstm_encoder(self):
        with tf.variable_scope("mt_lstm"):
            with tf.variable_scope("forwards"):
                forward_lstm_cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                                                                  for _ in range(self.layers)])
            with tf.variable_scope("backwards"):
                backward_lstm_cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                                                                   for _ in range(self.layers)])

            seq_len = tf.reduce_sum(tf.sign(tf.reduce_sum(self.encoded_sequence_placeholder, axis=2)), axis=1)
            fw_encoding, bw_encoding, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_lstm_cells,
                cell_bw=backward_lstm_cells,
                sequence_length=seq_len,
                inputs=self.encoded_sequence_placeholder,
                initial_state_fw=forward_lstm_cells.zero_state(self.batch_size, dtype=tf.float32),
                initial_state_bw=backward_lstm_cells.zero_state(self.batch_size, dtype=tf.float32)
            )

        return tf.concat([fw_encoding, bw_encoding], axis=2)

    def _run_decoder(self):
        decoder = NMTDecoder(encoder_outputs=self.encoded_sequence, translation_vocab_size=self.translation_vocab_size,
                             translation_embedding_matrix=self.translation_embedding_matrix, batch_size=self.batch_size,
                             sos_id=self.sos_id, eos_id=self.eos_id, decoder_hidden_size=self.decoder_output_size,
                             encoder_hidden_size=self.hidden_size)

        decoder_outputs, _, seq_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, inpute_finished=True,
                                                                            maximum_iterations=self.max_decoder_seq_len)
        return decoder_outputs, seq_lengths

    def _add_loss(self, translated_sequence):
        loss = tf.contrib.seq2seq.sequence_loss(logits=translated_sequence.rnn_output, targets=self.labels_placeholder,
                                                weights=tf.sign(self.labels_placeholder))
        return loss

    def _add_train_step(self, loss_op):
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_placeholder)
        return opt.minimize(loss_op)

