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
    for word in word_to_freq:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

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


def sample(docs, word_to_index, epoch_size, batch_size, q):
    for i in range(epoch_size):
        res = []
        p = np.random.permutation(len(docs))
        for j in range(0, len(p), batch_size):
            k = p[j:j + batch_size]
            max_len_1 = max(len(docs[k_][0]) for k_ in k)
            max_len_2 = max(len(docs[k_][1]) for k_ in k)
            X_doc_1_ = np.full([len(k), max_len_1], word_to_index['</sent>'])
            X_doc_2_ = np.full([len(k), max_len_2], word_to_index['</sent>'])
            mask_1_ = np.zeros([len(k), max_len_1], dtype=np.float32)
            mask_2_ = np.zeros([len(k), max_len_2], dtype=np.float32)
            y_ = [docs[k_][2] for k_ in k]
            for i_k, k_ in enumerate(k):
                doc_1 = [word_to_index[word] for word in docs[k_][0] if word in word_to_index]
                doc_2 = [word_to_index[word] for word in docs[k_][1] if word in word_to_index]
                X_doc_1_[i_k, :len(doc_1)] = doc_1
                X_doc_2_[i_k, :len(doc_2)] = doc_2
                mask_1_[i_k, :len(doc_1)] = 1
                mask_2_[i_k, :len(doc_2)] = 1
            res.append((X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_))
        q.put(res)


def attend_intra(w, emb, mask, n_intra_bias, long_dist_bias, dist_biases):
    i = tf.range(0, tf.shape(mask)[1], dtype=tf.int32)
    ij = tf.expand_dims(i, 1) - tf.expand_dims(i, 0)
    ij_mask = tf.cast(tf.logical_and(tf.less_equal(ij, n_intra_bias), tf.greater_equal(ij, -n_intra_bias)), tf.float32)
    w = w + (1 - ij_mask) * long_dist_bias + ij_mask * tf.gather(dist_biases, ij + n_intra_bias)
    mask = tf.expand_dims(mask, 1)
    norm = tf.nn.softmax(mask * w + (-1 / mask + 1))
    return tf.matmul(norm, emb)


def attend_inter(w, emb, mask):
    mask = tf.expand_dims(mask, 1)
    norm = tf.nn.softmax(mask * w + (-1 / mask + 1))
    return tf.matmul(norm, emb)


# noinspection PyTypeChecker
def run_sum(
    train, val, test, word_to_index, emb_size, emb_glove, context_size, n_proj, n_classif,
    dropout_rate, pred_thres, lr, batch_size, epoch_size
):
    # special words
    word_to_index['<sent>'] = len(word_to_index)
    word_to_index['</sent>'] = len(word_to_index)

    # network
    tf.reset_default_graph()
    X_doc_1 = tf.placeholder(tf.int32, [None, None])
    X_doc_2 = tf.placeholder(tf.int32, [None, None])
    mask_1 = tf.placeholder(tf.float32, [None, None])
    mask_2 = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool, [])

    emb_shape = [len(word_to_index), emb_size]
    emb = tf.Variable(tf.zeros(emb_shape) if emb_glove else tf.random_normal(emb_shape, 0, 0.01))

    sent = [None, None]
    l_proj = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_proj], [])
    for i in range(2):
        X_doc = [X_doc_1, X_doc_2][i]
        mask = [mask_1, mask_2][i]
        unigram = tf.nn.embedding_lookup(emb, X_doc)
        batch_size_, n_words_ = tf.shape(X_doc)[0], tf.shape(X_doc)[1]
        X_doc_start = tf.fill([batch_size_, context_size], word_to_index['<sent>'])
        X_doc_end = tf.fill([batch_size_, context_size], word_to_index['</sent>'])
        X_doc_n = tf.concat([X_doc_start, X_doc, X_doc_end], 1)
        X_doc_n = tf.map_fn(lambda j: X_doc_n[:, j:j + n_words_], tf.range(2 * context_size + 1))
        X_doc_n = tf.transpose(X_doc_n, [1, 0, 2])
        ngram = tf.nn.embedding_lookup(emb, X_doc_n)
        ngram = tf.reshape(ngram, [batch_size_, -1, (2 * context_size + 1) * emb_size])
        ngram = apply_layers(l_proj, ngram, training=training)
        sent[i] = tf.concat([
            tf.reduce_sum(tf.expand_dims(mask, -1) * unigram, 1),
            tf.reduce_sum(tf.expand_dims(mask, -1) * ngram, 1)
        ], 1)

    l_classif = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_classif], [])
    logits = apply_layers(l_classif, tf.concat(sent, 1), training=training)
    logits = tf.layers.dense(logits, 2, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    probs = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

    opt = tf.train.AdagradOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(loss)
    grads = [(grad, var) for grad, var in grads if var != emb] if emb_glove else grads
    train_op = opt.apply_gradients(grads)

    # run
    with tf.Session() as sess:
        # start sampling
        qs = {name: Queue(1) for name in ('train', 'val', 'test')}
        for name, docs in ('train', train), ('val', val), ('test', test):
            Process(target=sample, args=(docs, word_to_index, epoch_size, batch_size, qs[name])).start()

        # initialize variables
        sess.run(tf.global_variables_initializer())
        if emb_glove:
            emb_0 = tf.Variable(0., validate_shape=False)
            saver = tf.train.Saver({'emb': emb_0})
            saver.restore(sess, '__cache__/tf/emb/model.ckpt')
            sess.run(emb[:tf.shape(emb_0)[0]].assign(emb_0))

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            total_loss, correct = 0, {'val': 0, 'test': 0}
            for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in qs['train'].get():
                _, batch_loss = sess.run([train_op, loss], feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, y: y_, training: True,
                })
                total_loss += len(y_) * batch_loss
            for name in 'val', 'test':
                for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in qs[name].get():
                    probs_ = sess.run(probs, feed_dict={
                        X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, training: False,
                    })
                    correct[name] += np.sum((probs_[:, 1] >= pred_thres) == y_)
            print(
                datetime.datetime.now(),
                f'finished epoch {i}, loss: {total_loss / len(train):f}, '
                f'val acc: {correct["val"] / len(val):f}, test acc: {correct["test"] / len(test):f}'
            )


# noinspection PyTypeChecker
def run_decatt(
    train, val, test, word_to_index, intra_sent, emb_size, emb_glove, context_size,
    n_intra, n_intra_bias, n_attend, n_compare, n_classif, dropout_rate, pred_thres, lr, batch_size, epoch_size
):
    # special words
    word_to_index['<sent>'] = len(word_to_index)
    word_to_index['</sent>'] = len(word_to_index)

    # network
    tf.reset_default_graph()
    X_doc_1 = tf.placeholder(tf.int32, [None, None, 2 * context_size + 1])
    X_doc_2 = tf.placeholder(tf.int32, [None, None, 2 * context_size + 1])
    mask_1 = tf.placeholder(tf.float32, [None, None])
    mask_2 = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.int32, [None])
    training = tf.placeholder(tf.bool, [])

    emb_shape = [len(word_to_index), emb_size]
    emb = tf.Variable(tf.zeros(emb_shape) if emb_glove else tf.random_normal(emb_shape, 0, 0.01))

    ngram = [None, None]
    for i in range(2):
        X_doc = [X_doc_1, X_doc_2][i]
        batch_size_, n_words_ = tf.shape(X_doc)[0], tf.shape(X_doc)[1]
        X_doc_start = tf.fill([batch_size_, context_size], word_to_index['<sent>'])
        X_doc_end = tf.fill([batch_size_, context_size], word_to_index['</sent>'])
        X_doc_n = tf.concat([X_doc_start, X_doc, X_doc_end], 1)
        X_doc_n = tf.map_fn(lambda j: X_doc_n[:, j:j + n_words_], tf.range(2 * context_size + 1))
        X_doc_n = tf.transpose(X_doc_n, [1, 0, 2])
        ngram = tf.nn.embedding_lookup(emb, X_doc_n)
        ngram[i] = tf.reshape(ngram, [batch_size_, -1, (2 * context_size + 1) * emb_size])

    if intra_sent:
        l_intra = sum([[
            Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
            Dropout(rate=dropout_rate),
        ] for n in n_intra], [])
        long_dist_bias = tf.Variable(tf.zeros([]))
        dist_biases = tf.Variable(tf.zeros([2 * n_intra_bias + 1]))
        for i in range(2):
            intra_d = apply_layers(l_intra, ngram[i], training=training)
            intra_w = tf.matmul(intra_d, tf.transpose(intra_d, [0, 2, 1]))
            ngram[i] = tf.concat([ngram[i], attend_intra(
                intra_w, ngram[i], [mask_1, mask_2][i], n_intra_bias, long_dist_bias, dist_biases
            )], 2)

    l_attend = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_attend], [])
    attend_d_1 = apply_layers(l_attend, ngram[0], training=training)
    attend_d_2 = apply_layers(l_attend, ngram[1], training=training)
    attend_w = tf.matmul(attend_d_1, tf.transpose(attend_d_2, [0, 2, 1]))
    attend_1 = attend_inter(attend_w, ngram[1], mask_2)
    attend_2 = attend_inter(tf.transpose(attend_w, [0, 2, 1]), ngram[0], mask_1)

    l_compare = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_compare], [])
    compare_1 = apply_layers(l_compare, tf.concat([ngram[0], attend_1], 2), training=training)
    compare_2 = apply_layers(l_compare, tf.concat([ngram[1], attend_2], 2), training=training)

    agg_1 = tf.reduce_sum(tf.expand_dims(mask_1, -1) * compare_1, 1)
    agg_2 = tf.reduce_sum(tf.expand_dims(mask_2, -1) * compare_2, 1)
    l_classif = sum([[
        Dense(n, tf.nn.relu, kernel_initializer=init_ops.RandomNormal(0, 0.01)),
        Dropout(rate=dropout_rate),
    ] for n in n_classif], [])
    logits = apply_layers(l_classif, tf.concat([agg_1, agg_2], 1), training=training)
    logits = tf.layers.dense(logits, 2, kernel_initializer=init_ops.RandomNormal(0, 0.01))
    probs = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

    opt = tf.train.AdagradOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(loss)
    grads = [(grad, var) for grad, var in grads if var != emb] if emb_glove else grads
    train_op = opt.apply_gradients(grads)

    # run
    with tf.Session() as sess:
        # start sampling
        qs = {name: Queue(1) for name in ('train', 'val', 'test')}
        for name, docs in ('train', train), ('val', val), ('test', test):
            Process(target=sample, args=(docs, word_to_index, context_size, epoch_size, batch_size, qs[name])).start()

        # initialize variables
        sess.run(tf.global_variables_initializer())
        if emb_glove:
            emb_0 = tf.Variable(0., validate_shape=False)
            saver = tf.train.Saver({'emb': emb_0})
            saver.restore(sess, '__cache__/tf/emb/model.ckpt')
            sess.run(emb[:tf.shape(emb_0)[0]].assign(emb_0))

        # train
        print(datetime.datetime.now(), 'started training')
        for i in range(epoch_size):
            total_loss, correct = 0, {'val': 0, 'test': 0}
            for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in qs['train'].get():
                _, batch_loss = sess.run([train_op, loss], feed_dict={
                    X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, y: y_, training: True,
                })
                total_loss += len(y_) * batch_loss
            for name in 'val', 'test':
                for X_doc_1_, X_doc_2_, mask_1_, mask_2_, y_ in qs[name].get():
                    probs_ = sess.run(probs, feed_dict={
                        X_doc_1: X_doc_1_, X_doc_2: X_doc_2_, mask_1: mask_1_, mask_2: mask_2_, training: False,
                    })
                    correct[name] += np.sum((probs_[:, 1] >= pred_thres) == y_)
            print(
                datetime.datetime.now(),
                f'finished epoch {i}, loss: {total_loss / len(train):f}, '
                f'val acc: {correct["val"] / len(val):f}, test acc: {correct["test"] / len(test):f}'
            )


def main():
    print(' '.join(shlex.quote(arg) for arg in sys.argv[1:]))
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('method', choices=('sum', 'decatt'))
    arg_parser.add_argument('hyperparams')
    args = arg_parser.parse_args()
    train, val, test = load_data()
    word_to_index = gen_tables(train, val, test)
    train.extend([(doc[1], doc[0], doc[2]) for doc in train])
    if args.method == 'sum':
        run_sum(train, val, test, word_to_index, **json.loads(args.hyperparams))
    elif args.method == 'decatt':
        run_decatt(train, val, test, word_to_index, **json.loads(args.hyperparams))

if __name__ == '__main__':
    main()
