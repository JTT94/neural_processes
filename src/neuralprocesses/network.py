import tensorflow as tf

from . import GaussianParams, NeuralProcessParams


def encoder_h(context_xys: tf.Tensor, params: NeuralProcessParams) -> tf.Tensor:
    """Map context inputs (x_i, y_i) to r_i

    Creates a fully connected network with a single sigmoid hidden layer and linear output layer.

    Parameters
    ----------
    context_xys
        Input tensor, shape: (n_samples, dim_x + dim_y)
    params
        Neural process parameters

    Returns
    -------
        Output tensor of encoder network
    """
    hidden_layer = context_xys
    # First layers are relu
    for i, n_hidden_units in enumerate(params.n_hidden_units_h):
        hidden_layer = tf.layers.dense(hidden_layer, n_hidden_units,
                                       activation=tf.nn.relu,
                                       name='encoder_layer_{}'.format(i),
                                       reuse=tf.AUTO_REUSE,
                                       kernel_initializer='normal')

    # Last layer is simple linear
    i = len(params.n_hidden_units_h)
    r = tf.layers.dense(hidden_layer, params.dim_r,
                        name='encoder_layer_{}'.format(i),
                        reuse=tf.AUTO_REUSE,
                        kernel_initializer='normal')
    return r


def aggregate_r(context_rs: tf.Tensor) -> tf.Tensor:
    """Aggregate the output of the encoder to a single representation

    Creates an aggregation (mean) operator to combine the encodings of multiple context inputs

    Parameters
    ----------
    context_rs
        Input encodings tensor, shape: (n_samples, dim_r)

    Returns
    -------
        Output tensor of aggregation result
    """
    mean = tf.reduce_mean(context_rs, axis=0)

    r = tf.reshape(mean, [1, -1])
    return r


def get_z_params(context_r: tf.Tensor, params: NeuralProcessParams) -> GaussianParams:
    """Map encoding to mean and covariance of the random variable Z

    Creates a linear dense layer to map encoding to mu_z, and another linear mapping + a softplus activation for Sigma_z

    Parameters
    ----------
    context_r
        Input encoding tensor, shape: (1, dim_r)
    params
        Neural process parameters

    Returns
    -------
        Output tensors of the mappings for mu_z and Sigma_z
    """
    mu = tf.layers.dense(context_r, params.dim_z, name="z_params_mu", reuse=tf.AUTO_REUSE, kernel_initializer='normal')

    sigma = tf.layers.dense(context_r, params.dim_z, name="z_params_sigma", reuse=tf.AUTO_REUSE,
                            kernel_initializer='normal')
    sigma = tf.nn.softplus(sigma)

    return GaussianParams(mu, sigma)


def decoder_g(z_samples: tf.Tensor, input_xs: tf.Tensor, params: NeuralProcessParams,
              noise_std: float = 0.05) -> GaussianParams:
    """Determine output y* by decoding input and latent variable

    Creates a fully connected network with a single sigmoid hidden layer and linear output layer.

    Parameters
    ----------
    z_samples
        Random samples from the latent variable distribution, shape: (n_z_draws, dim_z)
    input_xs
        Input values to predict for, shape: (n_x_samples, dim_x)
    params
        Neural process parameters
    noise_std
        Constant standard deviation used on output

    Returns
    -------
        Output tensors for the parameters of Gaussian distributions for target outputy, where its mean mu has shape
        (n_x_samples, n_z_draws)
        TODO: this assumes/forces dim_y = 1
    """
    # inputs dimensions
    # z_sample has dim [n_draws, dim_z]
    # x_star has dim [N_star, dim_x]

    n_draws = z_samples.get_shape().as_list()[0]
    n_xs = tf.shape(input_xs)[0]

    # Repeat z samples for each x*
    z_samples_repeat = tf.expand_dims(z_samples, axis=1)
    z_samples_repeat = tf.tile(z_samples_repeat, [1, n_xs, 1])

    # Repeat x* for each z sample
    x_star_repeat = tf.expand_dims(input_xs, axis=0)
    x_star_repeat = tf.tile(x_star_repeat, [n_draws, 1, 1])

    # Concatenate x* and z
    # shape: (n_z_draws, n_xs, dim_x + dim_z)
    inputs = tf.concat([x_star_repeat, z_samples_repeat], axis=2)

    hidden_layer = inputs
    # First layers are relu
    for i, n_hidden_units in enumerate(params.n_hidden_units_g):
        hidden_layer = tf.layers.dense(hidden_layer, n_hidden_units,
                                       activation=tf.nn.relu,
                                       name='decoder_layer_{}'.format(i),
                                       reuse=tf.AUTO_REUSE,
                                       kernel_initializer='normal')

    # Last layer is simple linear
    i = len(params.n_hidden_units_g)
    hidden_layer = tf.layers.dense(hidden_layer, 1,
                                   name='decoder_layer_{}'.format(i),
                                   reuse=tf.AUTO_REUSE,
                                   kernel_initializer='normal')

    # mu will be of the shape [N_star, n_draws]
    mu_star = tf.squeeze(hidden_layer, axis=2)
    mu_star = tf.transpose(mu_star)

    sigma_star = tf.constant(noise_std, dtype=tf.float32)

    return GaussianParams(mu_star, sigma_star)


def xy_to_z_params(context_xs: tf.Tensor, context_ys: tf.Tensor,
                   params: NeuralProcessParams) -> GaussianParams:
    """Wrapper to create full network from context samples to parameters of pdf of Z

    Parameters
    ----------
    context_xs
        Tensor with context features, shape: (n_samples, dim_x)
    context_ys
        Tensor with context targets, shape: (n_samples, dim_y)
    params
        Neural process parameters

     encoder_h
        Neural process generator function

    Returns
    -------
        Output tensors of the mappings for mu_z and Sigma_z
    """
    xys = tf.concat([context_xs, context_ys], axis=1)
    rs = encoder_h(xys, params)
    r = aggregate_r(rs)
    z_params = get_z_params(r, params)

    return z_params
