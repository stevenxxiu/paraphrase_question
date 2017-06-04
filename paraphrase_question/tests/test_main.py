import unittest

import numpy as np
import tensorflow as tf
from numpy.testing import assert_almost_equal

from paraphrase_question.main import attend_inter, attend_intra


class TestAttend(unittest.TestCase):
    def test_attend_intra(self):
        emb = np.float32(np.random.uniform(-1, 1, (10, 10)))
        X_doc = np.int32([[0, 1, 2, 3, 4, 0, 0]])
        mask = np.float32([[1, 1, 1, 1, 1, 0, 0]])
        n_intra_bias = 1
        w = np.float32(np.random.uniform(-1, 1, (1, 7, 7)))
        long_dist_bias = np.float32(np.random.uniform(-1, 1))
        dist_biases = np.float32(np.random.uniform(-1, 1, 2 * n_intra_bias + 1))
        with tf.Session() as sess:
            emb_ = tf.nn.embedding_lookup(emb, X_doc)
            intra = sess.run(attend_intra(w, emb_, mask, n_intra_bias, long_dist_bias, dist_biases))

            def get_bias(i_, j_):
                return dist_biases[i_ - j_ + n_intra_bias] if np.abs(i_ - j_) <= n_intra_bias else long_dist_bias

            for i in range(5):
                assert_almost_equal(np.sum(np.exp(w[0, i, j] + get_bias(i, j)) / np.sum(
                    np.exp(w[0, i, k] + get_bias(i, k)) for k in range(5)
                ) * emb[X_doc[0, j]] for j in range(5)), intra[0, i], decimal=6)

    def test_attend_inter(self):
        emb = np.float32(np.random.uniform(-1, 1, (10, 10)))
        X_doc_1 = np.int32([[0, 1, 2, 3, 4, 0, 0]])
        X_doc_2 = np.int32([[0, 1, 2, 3, 4, 5, 0]])
        mask_1 = np.float32([[1, 1, 1, 1, 1, 0, 0]])
        mask_2 = np.float32([[1, 1, 1, 1, 1, 1, 0]])
        w = np.float32(np.random.uniform(-1, 1, (1, 7, 7)))
        with tf.Session() as sess:
            emb_1 = tf.nn.embedding_lookup(emb, X_doc_1)
            emb_2 = tf.nn.embedding_lookup(emb, X_doc_2)
            attend_1 = sess.run(attend_inter(w, emb_2, mask_2))
            attend_2 = sess.run(attend_inter(tf.transpose(w, [0, 2, 1]), emb_1, mask_1))
            for i in range(5):
                assert_almost_equal(np.sum(np.exp(w[0, i, j]) / np.sum(
                    np.exp(w[0, i, k]) for k in range(6)
                ) * emb[X_doc_2[0, j]] for j in range(6)), attend_1[0, i], decimal=6)
            for j in range(6):
                assert_almost_equal(np.sum(np.exp(w[0, i, j]) / np.sum(
                    np.exp(w[0, k, j]) for k in range(5)
                ) * emb[X_doc_1[0, i]] for i in range(5)), attend_2[0, j], decimal=6)
