from collections import namedtuple
<<<<<<< HEAD

NeuralProcessParams = namedtuple('NeuralProcessParams', ['dim_r', 'dim_z', 'n_hidden_units_h', 'n_hidden_units_g'])
=======
import random
import tensorflow as tf
import tensorflow as tf
import numpy as np
from typing import Dict

NeuralProcessParams = namedtuple('NeuralProcessParams', ['dim_r', 'dim_z', 'n_hidden_units_h', 'n_hidden_units_g', 'learning_rate'])
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
GaussianParams = namedtuple('GaussianParams', ['mu', 'sigma'])

# function to calculate the log likelihood for Gaussians
def getLoglik(params, data):
    norm_dist = tf.distributions.Normal(loc=params.mu, scale=params.sigma)  # define dist
    loglik = norm_dist.log_prob(data)  # calculate prob of data
    loglik = tf.reduce_sum(loglik, axis=0)  # sum loglik components
    loglik = tf.reduce_mean(loglik)  # average across samples

    return loglik


# KL of gaussians https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
def getKLdist(p_mu, p_sigma, q_mu, q_sigma):
    p_sigma_sq = tf.square(p_sigma) + 1e-16
    q_sigma_sq = tf.square(q_sigma) + 1e-16

<<<<<<< HEAD
    KLdist = (q_sigma / p_sigma) + tf.square(p_mu - q_mu) / p_sigma_sq - 1.0 + tf.log(p_sigma_sq / q_sigma_sq + 1e-16)
=======
    KLdist = q_sigma_sq / p_sigma_sq + tf.square(q_mu - p_mu) / q_sigma_sq - 1.0 + tf.log(q_sigma_sq / p_sigma_sq + 1e-16)
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
    KLdist = 0.5 * tf.reduce_sum(KLdist)

    return KLdist

<<<<<<< HEAD
def get_context_target(X, Y, n_context):
  ind = set(range(xs.shape[0]))
  ind_c = set(random.sample(ind, n_context))
  ind_t = ind - ind_c
  X_context = X[list(ind_c)]
  Y_context = Y[list(ind_c)]
  X_target = X[list(ind_t)]
  Y_target = Y[list(ind_c)]

  return X_context, Y_context, X_target, Y_target
=======



def split_context_target(xs: np.array, ys: np.array, n_context: int,
                         context_xs: tf.Tensor, context_ys: tf.Tensor, target_xs: tf.Tensor, target_ys: tf.Tensor) \
        -> Dict[tf.Tensor, np.array]:
    """Split samples randomly into context and target sets
    """
    indices = set(range(ys.shape[0]))
    context_set_indices = set(random.sample(indices, n_context))
    target_set_indices = indices - context_set_indices

    return {
        context_xs: xs[list(context_set_indices), :],
        context_ys: ys[list(context_set_indices), :],
        target_xs: xs[list(target_set_indices), :],
        target_ys: ys[list(target_set_indices), :]
    }





>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
