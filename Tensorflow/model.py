import tensorflow as tf
import numpy as np
import json

import data_helpers
from configure import FLAGS
from utils import Two2Three


def attention(inputs):
    # Trainable parameters
    hidden_size = inputs.shape[2].value
    u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

    with tf.name_scope('v'):
        v = tf.tanh(inputs)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    # Final output with tanh
    output = tf.tanh(output)

    return output, alphas


class AttLSTM:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 hidden_size, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_position = tf.placeholder(tf.int32, shape=[None, 2], name='input_position')
        self.input_y = tf.placeholder(tf.int64, shape=[None], name='input_y')
        self.mask = tf.placeholder(tf.int32, shape=[1, sequence_length], name='position_mask')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal

        # Word Embedding Layer
        with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.emb_dropout_keep_prob)

        # Position Feature
        with tf.variable_scope('position-features'):
            self.pf = self.mask[:,:,None] - self.input_position[:,None,:]

        # Bidirectional LSTM
        with tf.variable_scope("bi-lstm"):
            self.embedded_input = tf.concat([self.embedded_chars, tf.to_float(self.pf)], axis=2)
            _fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.rnn_dropout_keep_prob)
            _bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.rnn_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                  cell_bw=bw_cell,
                                                                  inputs=self.embedded_input,
                                                                  sequence_length=self._length(self.input_text),
                                                                  dtype=tf.float32)
            self.rnn_outputs = tf.add(self.rnn_outputs[0], self.rnn_outputs[1])

        # Attention
        with tf.variable_scope('attention'):
            self.attn, self.alphas = attention(self.rnn_outputs)

        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.attn, self.dropout_keep_prob)

        # Fully connected layer
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            self.probabilities = tf.nn.softmax(self.logits, name="probabilities")[:, 1]

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length


def predict(data, params_path=FLAGS.checkpoint_dir):
    num_sent = len(data)
    mask = np.ones(shape=[FLAGS.sequence_length]).nonzero()
    with tf.device('/cpu:0'):
        x_text_ta, x_position_ta, x_id_ta, x_text_av, x_position_av, x_id_av = data_helpers.load_test_data(data)

    checkpoint_file = params_path + '/'

    # find Time & Attribute tuples
    checkpoint_file_ta = checkpoint_file + 'model_ta_final'
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file_ta))
            saver.restore(sess, checkpoint_file_ta)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_position = graph.get_operation_by_name("input_position").outputs[0]
            input_mask = graph.get_operation_by_name("position_mask").outputs[0]
            emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
            rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            probablities = graph.get_operation_by_name("output/probabilities").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(x_text_ta, x_position_ta, None, FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            preds = []
            probs = []
            for x_batch in batches:
                x_text_batch, x_position_batch = x_batch
                pred, prob = sess.run([predictions, probablities], {input_text: x_text_batch,
                                                                    input_position: x_position_batch,
                                                                    input_mask: mask,
                                                                    emb_dropout_keep_prob: 1.0,
                                                                    rnn_dropout_keep_prob: 1.0,
                                                                    dropout_keep_prob: 1.0})
                preds.append(pred)
                probs.append(prob[:, 1])
            preds = np.concatenate(preds)
            probs = np.concatenate(probs)
            time_attr = np.concatenate((x_position_ta, probs[:, None]), axis=1)
            mask_ta = np.where(preds == 1)
            id_ta = x_id_ta[mask_ta].copy()
            time_attr = time_attr[mask_ta].copy()

    # find Attribute & Value tuples
    checkpoint_file_av = checkpoint_file + 'model_av_final'
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file_av))
            saver.restore(sess, checkpoint_file_av)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_position = graph.get_operation_by_name("input_position").outputs[0]
            input_mask = graph.get_operation_by_name("position_mask").outputs[0]
            emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
            rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            probablities = graph.get_operation_by_name("output/probabilities").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(x_text_av, x_position_av, None, FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            preds = []
            probs = []
            for x_batch in batches:
                x_text_batch, x_position_batch = x_batch
                pred, prob = sess.run([predictions, probablities], {input_text: x_text_batch,
                                                                    input_position: x_position_batch,
                                                                    input_mask: mask,
                                                                    emb_dropout_keep_prob: 1.0,
                                                                    rnn_dropout_keep_prob: 1.0,
                                                                    dropout_keep_prob: 1.0})
                preds.append(pred)
                probs.append(prob[:, 1])
            preds = np.concatenate(preds)
            probs = np.concatenate(probs)
            attr_value = np.concatenate((x_position_av, probs[:, None]), axis=1)
            mask_av = np.where(preds == 1)
            id_av = x_id_av[mask_av].copy()
            attr_value = attr_value[mask_av].copy()

    # combining (time, attribute, value) tuples
    two_tuples = []
    for id in range(num_sent):
        mask_ta = np.where(id_ta == id)
        mask_av = np.where(id_av == id)
        two_tuples.append({'time_attr': time_attr[mask_ta], 'attr_val': attr_value[mask_av]})

    results = []
    for item in two_tuples:
        results.append(Two2Three(item))

    for id in range(num_sent):
        data[id]['results'] = results[id].tolist()

    return data


def main():
    data = json.load(open('data_2d_tuple/source_data/assignment_training_data_word_segment.json', 'rb'))
    global test
    test = predict(data)


if __name__ == "__main__":
    main()