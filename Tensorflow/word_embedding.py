import numpy as np
import json
import os
import pickle
import collections
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def read_voc(filepath):
    f = pickle.load(open(filepath, 'rb'))
    idx2word = f['idx2word']
    return idx2word


def read_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def get_sentence(data):
    sentence = []
    for idx, entry in enumerate(data):
        sentence.append(entry['indexes'])
    return sentence


voc_frame = 'voc-d.pkl'
data_frame = 'assignment_training_data_word_segment.json'
embedding_frame = 'embedding.npy'
idx2word = read_voc(voc_frame)
voc_size = len(idx2word)
dataset = read_data(data_frame)
sentset = get_sentence(dataset)

embedding_size = 100
batch_size = 100
skip_window = 2
num_skips = 2
num_sampled = 64

sent_index = [0] * len(sentset)


def generate_batch(sentset_idx, batch_size, num_skips, skip_window):
    global sent_index
    sent = sentset[sentset_idx]
    idx = sent_index[sentset_idx]
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buf = collections.deque(maxlen=span)
    for _ in range(span):
        # set_trace()
        # print(sent)
        buf.append(sent[idx])
        idx = (idx + 1) % len(sent)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buf[skip_window]
            labels[i * num_skips + j, 0] = buf[target]
        buf.append(sent[idx])
        idx = (idx + 1) % len(sent)
    sent_index[sentset_idx] = idx
    return batch, labels


# generate_batch(0,batch_size,num_skips,skip_window)
graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([voc_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([voc_size]))
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=voc_size))
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

num_steps = 5000000
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(step % len(sentset), batch_size, num_skips, skip_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 100000 == 0 and step > 0:
            print('Average loss at step %d: %f' % (step, average_loss / 100000))
            average_loss = 0
    word2vec = normalized_embeddings.eval()

np.save(embedding_frame, word2vec)
