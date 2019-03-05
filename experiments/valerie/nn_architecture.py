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
                                       )

    # create final layer that maps to rs
    i = len(params.n_hidden_units_h)
    rs = tf.layers.dense(hidden_layer
                         , units=params.dim_r
                         , name='encoder_layer_{}'.format(i)
                         , reuse=tf.AUTO_REUSE
                         )
    return rs


def aggregate_r(rs):
    mean_r = tf.reduce_mean(rs, axis=0)
    r = tf.reshape(mean_r, [1, -1])
    return r


## specific to GP prior
def get_z_params(r, params):
    mu = tf.layers.dense(r
                         , units=params.dim_z
                         , name='z_params_mu'
                         , reuse=tf.AUTO_REUSE)

    sigma = tf.layers.dense(r
                            , units=params.dim_z
                            , name='z_params_sigma'
                            , reuse=tf.AUTO_REUSE)

    sigma = tf.nn.softplus(sigma)

    return GaussianParams(mu, sigma)


def decoder_g(z_samples, x_target, params, noise_std):
    # generate y_pred_params using neural net with z_samples and x_target as inputs

    # need to stack n_draws copies of x* together to concat with z  shape:  [n_draws, N_star, dim_x]
    n_draws = z_samples.get_shape().as_list()[0]  # n samples from z
    n_star = tf.shape(x_target)[0]  # n target xs

    z_star = tf.expand_dims(z_samples, axis=1)
    z_star = tf.tile(z_star, [1, n_star, 1])

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
                                       , reuse=tf.AUTO_REUSE)

    # last layer is simple linear layer: shape (5, ?, 1)
    i = len(params.n_hidden_units_g)
    hidden_layer = tf.layers.dense(hidden_layer
                                   , units=1
                                   , name='decoder_layer{}'.format(i)
                                   , reuse=tf.AUTO_REUSE
                                   )

    # drop last dim of the layer and transpose to get shape [n_star, n_draws]
    mu_star = tf.squeeze(hidden_layer, axis=2)
    mu_star = tf.transpose(mu_star)

    # constant noise - from a parameter
    sigma_star = tf.constant(noise_std, dtype=tf.float32)

    return GaussianParams(mu_star, sigma_star)


def map_xy_to_z(x, y, params):
    # concatenate X and Y to be the (x_i, y_i) input pair --> r_i
    xys = tf.concat([x, y], axis=1)

    rs = encoder_h(xys, params)
    r = aggregate_r(rs)
    z_params = get_z_params(r, params)

    return z_params
