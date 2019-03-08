<<<<<<< HEAD
=======
import tensorflow as tf
from functions import *

>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
def encoder_h(xys, params):
    # initialize hidden layer
    hidden_layer = xys

    # for the number of layers specified in params, create new hidden layer with correct number of neurons
    for i, n_units in enumerate(params.n_hidden_units_h):
        hidden_layer = tf.layers.dense(hidden_layer
                                       , units=n_units
                                       , activation=tf.nn.relu
                                       , name='encoder_layer_{}'.format(i)
                                       , reuse=tf.AUTO_REUSE  # don't reuse weight and bias variables across layers
<<<<<<< HEAD
                                       )
=======
                                       , kernel_initializer='normal')
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1

    # create final layer that maps to rs
    i = len(params.n_hidden_units_h)
    rs = tf.layers.dense(hidden_layer
                         , units=params.dim_r
                         , name='encoder_layer_{}'.format(i)
                         , reuse=tf.AUTO_REUSE
<<<<<<< HEAD
                         )
    return rs


=======
                         , kernel_initializer='normal')
    return rs



>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
def aggregate_r(rs):
    mean_r = tf.reduce_mean(rs, axis=0)
    r = tf.reshape(mean_r, [1, -1])
    return r


<<<<<<< HEAD
## specific to GP prior
=======
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
def get_z_params(r, params):
    mu = tf.layers.dense(r
                         , units=params.dim_z
                         , name='z_params_mu'
<<<<<<< HEAD
                         , reuse=tf.AUTO_REUSE)
=======
                         , reuse=tf.AUTO_REUSE
                         , kernel_initializer='normal')
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1

    sigma = tf.layers.dense(r
                            , units=params.dim_z
                            , name='z_params_sigma'
<<<<<<< HEAD
                            , reuse=tf.AUTO_REUSE)
=======
                            , reuse=tf.AUTO_REUSE
                            , kernel_initializer='normal')
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1

    sigma = tf.nn.softplus(sigma)

    return GaussianParams(mu, sigma)


def decoder_g(z_samples, x_target, params, noise_std):
    # generate y_pred_params using neural net with z_samples and x_target as inputs

<<<<<<< HEAD
    # need to stack n_draws copies of x* together to concat with z  shape:  [n_draws, N_star, dim_x]
    n_draws = z_samples.get_shape().as_list()[0]  # n samples from z
    n_star = tf.shape(x_target)[0]  # n target xs

    z_star = tf.expand_dims(z_samples, axis=1)
    z_star = tf.tile(z_star, [1, n_star, 1])

=======
    n_draws = z_samples.get_shape().as_list()[0]  # n samples from z
    n_star = tf.shape(x_target)[0]  # n target xs

    # need to stack n_draws copies of x* together to concat with z  shape:  [n_draws, n_star, dim_z]
    z_star = tf.expand_dims(z_samples, axis=1)
    z_star = tf.tile(z_star, [1, n_star, 1])

    # need to stack n_draws copies of x* together to concat with z  shape:  [n_draws, n_star, dim_x]
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
    x_star = tf.expand_dims(x_target, axis=0)
    x_star = tf.tile(x_star, [n_draws, 1, 1])

    xzs_star = tf.concat([x_star, z_star], axis=2)  # shape: [n_draws, n_star, dim_x + dim_z]

    hidden_layer = xzs_star

    # build number of hidden ReLU layers specified in params.n_hidden_units_g
    for i, units in enumerate(params.n_hidden_units_g):
        hidden_layer = tf.layers.dense(hidden_layer
                                       , units=units
                                       , activation=tf.nn.relu
                                       , name='decoder_layer{}'.format(i)
<<<<<<< HEAD
                                       , reuse=tf.AUTO_REUSE)
=======
                                       , reuse=tf.AUTO_REUSE
                                       , kernel_initializer='normal')
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1

    # last layer is simple linear layer: shape (5, ?, 1)
    i = len(params.n_hidden_units_g)
    hidden_layer = tf.layers.dense(hidden_layer
                                   , units=1
                                   , name='decoder_layer{}'.format(i)
                                   , reuse=tf.AUTO_REUSE
<<<<<<< HEAD
                                   )
=======
                                   , kernel_initializer='normal')
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1

    # drop last dim of the layer and transpose to get shape [n_star, n_draws]
    mu_star = tf.squeeze(hidden_layer, axis=2)
    mu_star = tf.transpose(mu_star)

    # constant noise - from a parameter
    sigma_star = tf.constant(noise_std, dtype=tf.float32)

<<<<<<< HEAD
=======
    #sigma_star = tf.

>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
    return GaussianParams(mu_star, sigma_star)


def map_xy_to_z(x, y, params):
    # concatenate X and Y to be the (x_i, y_i) input pair --> r_i
    xys = tf.concat([x, y], axis=1)

    rs = encoder_h(xys, params)
    r = aggregate_r(rs)
    z_params = get_z_params(r, params)

    return z_params
