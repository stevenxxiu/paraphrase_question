import argparse
import csv
import datetime
import inspect
import json
import os
import shlex
import sys
from collections import defaultdict
from multiprocessing import Process, Queue

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.layers.core import Dense, Dropout
from tensorflow.python.ops import init_ops

memory = joblib.Memory('__cache__', verbose=0)


def load_data():
    res = []
    for file in ('train.tsv', 'dev.tsv', 'test.tsv'):
        with open(f'../../data/quora_question_pair/{file}', 'r', encoding='utf-8') as sr:
            cur_res = []
            for line in sr:
                label, sent_1, sent_2, _id = line.split('\t')
                cur_res.append((sent_1.split(' '), sent_2.split(' '), int(label)))
            res.append(cur_res)
    return res


@memory.cache(ignore=['train', 'val', 'test'])
def gen_tables(train, val, test):
    # load required glove vectors
    word_to_freq = defaultdict(int)
    for docs in train, val, test:
        for sent1, sent2, label in docs:
            for sent in sent1, sent2:
                for word in sent:
                    word_to_freq[word] += 1
    vecs = []
    word_to_index = {}
    with open('../../data/glove/glove.840B.300d.txt', 'r', encoding='utf-8') as sr:
        for line in sr:
            words = line.split(' ')
            if words[0] in word_to_freq:
                vecs.append(np.array(list(map(np.float32, words[1:]))))
                word_to_index[words[0]] = len(word_to_index)

    # visualize embeddings
    path = '__cache__/tf/emb'
    os.makedirs(path, exist_ok=True)
    config = projector.ProjectorConfig()
    emb_conf = config.embeddings.add()
    emb_conf.tensor_name = 'emb'
    emb_conf.metadata_path = os.path.abspath(os.path.join(path, 'emb_metadata.tsv'))
    with open(emb_conf.metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['Word', 'Frequency'])
        words = len(word_to_index) * [None]
        for word, i in word_to_index.items():
            words[i] = word
        for word in words:
            writer.writerow([word, word_to_freq[word]])
    summary_writer = tf.summary.FileWriter(path)
    projector.visualize_embeddings(summary_writer, config)
    emb = tf.Variable(tf.constant(np.array(vecs)))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver({'emb': emb})
        saver.save(sess, os.path.join(path, 'model.ckpt'), write_meta_graph=False)

    return word_to_index


def apply_layers(layers, input_, **kwargs):
    for layer in layers:
        names = inspect.signature(layer.call).parameters
        input_ = layer.apply(input_, **{name: arg for name, arg in kwargs.items() if name in names})
    return input_


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def sample(docs, word_to_index, context_size, epoch_size, batch_size, q):
    for i in range(epoch_size):
        res = []
        p = np.random.permutation(len(docs))
        for j in range(0, len(p), batch_size):
            k = p[j:j + batch_size]
            max_len_1 = max(len(docs[k_][0]) for k_ in k)
            max_len_2 = max(len(docs[k_][1]) for k_ in k)
            X_doc_1_ = np.zeros([len(k), max_len_1, context_size * 2 + 1], dtype=np.int32)
            X_doc_2_ = np.zeros([len(k), max_len_2, context_size * 2 + 1], dtype=np.int32)
            mask_1_ = np.zeros([len(k), max_len_1], dtype=np.float32)
            mask_2_ = np.zeros([len(k), max_len_2], dtype=np.float32)
            y_ = [docs[k_][2] for k_ in k]
            for i_k, k_ in enumerate(k):
                doc_1 = [word_to_index[word] for word in docs[k_][0] if word in word_to_index]
                doc_2 = [word_to_index[word] for word in docs[k_][1] if word in word_to_index]
                doc_1 = doc_1 or [np.random.randint(len(word_to_index))]
                doc_2 = doc_2 or [np.random.randint(len(word_to_index))]
                X_doc_1_[i_k, :len(doc_1)] = rolling_window(
                    np.pad(doc_1, context_size, 'constant', constant_values=word_to_index['\0']), context_size * 2 + 1
                )
                X_doc_2_[i_k, :len(doc_2)] = rolling_window(
                    np.pad(doc_2, context_size, 'constant', constant_values=word_to_index['\0']), context_size * 2 + 1
                )
                mask_1_[i_k, :len(doc_1)] = 1
                mask_2_[i_k, :len(doc_2)] = 1
            res.append((X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_))
        q.put(res)


def attend_intra(w, emb, mask, n_intra_bias, long_dist_bias, dist_biases, batch_size_):
    i = tf.range(0, tf.shape(mask)[1], dtype=tf.int32)
    ij = tf.reshape(i, [-1, 1]) - tf.reshape(i, [1, -1])
    ij_mask = tf.cast(tf.logical_and(tf.less_equal(ij, n_intra_bias), tf.greater_equal(ij, -n_intra_bias)), tf.float32)
    w = w + (1 - ij_mask) * long_dist_bias + ij_mask * tf.gather(dist_biases, ij + n_intra_bias)
    mask = tf.reshape(mask, [batch_size_, 1, -1])
    norm = tf.nn.softmax(mask * w + (-1 / mask + 1))
    return tf.matmul(norm, emb)


def attend_inter(w, emb, mask, batch_size_):
    mask = tf.reshape(mask, [batch_size_, 1, -1])
    norm = tf.nn.softmax(mask * w + (-1 / mask + 1))
    return tf.matmul(norm, emb)


# noinspection PyTypeChecker
def run_model(
    train, val, test, word_to_index, intra_sent, emb_size, context_size,
    n_intra, n_intra_bias, n_attend, n_compare, n_classif, dropout_rate, lr, batch_size, epoch_size
):
    # special words
    word_to_index['\0'] = len(word_to_index)

    # network
    tf.reset_default_graph()
    X_doc_1 = tf.placeholder(tf.int32, [None, None, None])
    X_doc_2 = tf.placeholder(tf.int32, [None, None, None])
    mask_1 = tf.placeholder(tf.float32, [None, None])
    mask_2 = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool, [])
    batch_size_ = tf.shape(X_doc_1)[0]

    emb = tf.Variable(tf.random_normal([len(word_to_index), emb_size], 0, 1))
    emb_ = [tf.reshape(
        tf.nn.embedding_lookup(emb, [X_doc_1, X_doc_2][i]), [batch_size_, -1, (2 * context_size + 1) * emb_size]
    ) for i in range(2)]
    if intra_sent:
        l_intra = sum([[
            Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
            Dropout(rate=dropout_rate),
        ] for n in n_intra], [])
        long_dist_bias = tf.Variable(tf.zeros([]))
        dist_biases = tf.Variable(tf.zeros([2 * n_intra_bias + 1]))
        for i in range(2):
            intra_d = apply_layers(l_intra, emb_[i], training=training)
            intra_w = tf.matmul(intra_d, tf.transpose(intra_d, [0, 2, 1]))
            emb_[i] = tf.concat([emb_[i], attend_intra(
                intra_w, emb_[i], [mask_1, mask_2][i], n_intra_bias, long_dist_bias, dist_biases, batch_size_
            )], 2)

    l_attend = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_attend], [])
    attend_d_1 = apply_layers(l_attend, emb_[0], training=training)
    attend_d_2 = apply_layers(l_attend, emb_[1], training=training)
    attend_w = tf.matmul(attend_d_1, tf.transpose(attend_d_2, [0, 2, 1]))
    attend_1 = attend_inter(attend_w, emb_[1], mask_2, batch_size_)
    attend_2 = attend_inter(tf.transpose(attend_w, [0, 2, 1]), emb_[0], mask_1, batch_size_)

    l_compare = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_compare], [])
    compare_1 = apply_layers(l_compare, tf.concat([emb_[0], attend_1], 2), training=training)
    compare_2 = apply_layers(l_compare, tf.concat([emb_[1], attend_2], 2), training=training)

    agg_1 = tf.reduce_sum(tf.reshape(mask_1, [batch_size_, -1, 1]) * compare_1, 1)
    agg_2 = tf.reduce_sum(tf.reshape(mask_2, [batch_size_, -1, 1]) * compare_2, 1)
    l_classif = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_classif], [])
    logits = apply_layers(l_classif, tf.concat([agg_1, agg_2], 1), training=training)
    logits = tf.layers.dense(logits, 3, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

    opt = tf.train.AdagradOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(loss)
    train_op = opt.apply_gradients([(grad, var) for grad, var in grads if var != emb])
    # train_op = opt.minimize(loss)

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # start sampling
        q_train, q_valid, q_test = Queue(1), Queue(1), Queue(1)
        Process(target=sample, args=(train, word_to_index, context_size, epoch_size, batch_size, q_train)).start()
        Process(target=sample, args=(val, word_to_index, context_size, epoch_size, batch_size, q_valid)).start()
        Process(target=sample, args=(test, word_to_index, context_size, epoch_size, batch_size, q_test)).start()

        # load pretrained word embeddings
        emb_0 = tf.Variable(0., validate_shape=False)
        saver = tf.train.Saver({'emb': emb_0})
        saver.restore(sess, '__cache__/tf/emb/model.ckpt')
        sess.run(emb[:tf.shape(emb_0)[0]].assign(emb_0))

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            total_loss, val_correct, test_correct = 0, 0, 0
            for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in q_train.get():
                _, batch_loss = sess.run([train_op, loss], feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, y: y_, training: True,
                })
                total_loss += len(y_) * batch_loss
            for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in q_valid.get():
                logits_ = sess.run(logits, feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, training: False,
                })
                val_correct += np.sum(np.argmax(logits_, 1) == y_)
            for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in q_test.get():
                logits_ = sess.run(logits, feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, training: False,
                })
                test_correct += np.sum(np.argmax(logits_, 1) == y_)
            print(
                datetime.datetime.now(),
                f'finished epoch {i}, loss: {total_loss / len(train):f}, '
                f'val acc: {val_correct / len(val):f}, test acc: {test_correct / len(test):f}'
            )


def main():
    print(' '.join(shlex.quote(arg) for arg in sys.argv[1:]))
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('hyperparams')
    args = arg_parser.parse_args()
    train, val, test = load_data()
    word_to_index = gen_tables(train, val, test)
    train.extend([(doc[1], doc[0], doc[2]) for doc in train])
    run_model(train, val, test, word_to_index=word_to_index, **json.loads(args.hyperparams))

if __name__ == '__main__':
    main()
