
<<<<<<< HEAD
# define wrapper for whole process

def NeuralProcess(x_context, y_context, x_target, y_target, learning_rate, params, n_draws, noise_std=0.05):
    # params: dim_r, dim_z, n_hidden_units_h, n_hidden_layers_g

    # reset and initialize new tensorflow session
    tf.reset_default_graph()
    sess = tf.Session()
=======
from nn_architecture import *
import tensorflow as tf

# define wrapper for whole process

def NeuralProcess(x_context, y_context, x_target, y_target, params, n_draws, noise_std=0.05):
    # params: dim_r, dim_z, n_hidden_units_h, n_hidden_layers_g

>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1

    # create combined x and y vectors
    x_all = tf.concat([x_context, x_target], axis=0)
    y_all = tf.concat([y_context, y_target], axis=0)

    #### Step 1: Encode context in latent variable representation
    z_params_context = map_xy_to_z(x_context, y_context, params)
    z_params_all = map_xy_to_z(x_all, y_all, params)

    #### Step 2: sample Z
    # seed z_samples with draw from standard normal
<<<<<<< HEAD
    epsilon = tf.random_normal(shape=[n_draws, params.z_dim])
=======
    epsilon = tf.random_normal(shape=[n_draws, params.dim_z])
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
    # scale noise using z_all.sigma
    z_samples = tf.multiply(epsilon, z_params_all.sigma)
    # re-center samples at z_all.mu
    z_samples = tf.add(z_samples, z_params_all.mu)

    #### Step 3: map sampled Zs and target Xs to Ys
    y_pred_params = decoder_g(z_samples, x_target, params, noise_std)

    #### Step 4: Calculate ELBO
    loglik = getLoglik(y_pred_params, y_target)
    KLdist = getKLdist(z_params_all.mu, z_params_all.sigma, z_params_context.mu, z_params_context)
    loss = tf.negative(loglik) + KLdist

    #### Step 5: Maximize ELBO, minimize loss
<<<<<<< HEAD
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
=======
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
    train_opt = optimizer.minimize(loss)

    return train_opt, loss


<<<<<<< HEAD
=======
def initializeNP(params):
    tf.reset_default_graph()
    sess = tf.Session()

    # Placeholders for training inputs
    x_context = tf.placeholder(tf.float32, (None, 1))
    y_context = tf.placeholder(tf.float32, (None, 1))
    x_target = tf.placeholder(tf.float32, (None, 1))
    y_target = tf.placeholder(tf.float32, (None, 1))

    train_op, loss = NeuralProcess(x_context=x_context, y_context=y_context
                                   , x_target=x_target, y_target=y_target
                                   , params=params
                                   , n_draws=7)

    # Initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess
>>>>>>> ac517c15662a2cefbb0bb6c45b3bfb4b5cd3bff1
