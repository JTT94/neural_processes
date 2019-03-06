from

def predictNP(x_target
              , x_context = None, y_context = None  #optionally supply context
              , epsilon = None                      #optionally supply randomness
              , params
              , n_draws = 1
              , noise_std = 0.05):

    # create tensors
    x_star = tf.constant(x_target, dtype=tf.float32)

    if y_context is not None:
        y = tf.constant(y_context, dtype=tf.float32)

    if x_context is not None:
        x = tf.constant(x_context, dtype=tf.float32)


    #if no epsilon provided, draw from standard normal with shape (n_draws, dim_z)
    if epsilon is None:
        epsilon = tf.random_normal((n_draws, params.dim_z))

    # if we're doing posterior prediction (i.e. have context)
    if x_context is not None and y_context is not None:
        #map (x_i, y_i) -> z
        z_params = map_xy_to_z(x = x_context, y = y_context, params = params)

        # use z params to adjust epsilon
        z_samples = tf.multiply(epsilon, z_params.sigma)
        z_samples = tf.add(z_samples, z_params.mu)
    else:
        z_samples = epsilon

    # use decoder_g (x_i, z_i) -> y_i
    y_params = decoder_g(z_samples, x_star, params = params, noise_std)

    return y_params
