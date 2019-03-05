from NP_demo import *

def NP_train(x_train, y_train, dim_struc, n_z_samples, n_epochs):
    """
    @param: x_train, y_train - dictionaries contian stuff for different tasks
    @param: dim_struc - dimensionality structure of our networks
    @param: n_z_samples - number of z sampels
    @param: n_epochs - number of training
    """

    # setup neural net structure
    r_dim, z_dim, x_dim, y_dim = dim_struc

    # setup coders
    r_encoder = REncoder(x_dim+y_dim, r_dim)
    z_encoder = ZEncoder(r_dim, z_dim)
    decoder = Decoder(x_dim+z_dim, y_dim)

    # setup optimiser
    opt = torch.optim.Adam(list(decoder.parameters())
                        +list(z_encoder.parameters())
                        +list(r_encoder.parameters()), 1e-3)

    # setup training
    losses = []
    z_param_dict = dict()
    task_ls = ["task " + str(i) for i in range(20)]

    for task in task_ls:
        for t in range(n_epochs):
            opt.zero_grad()

            # data gen
            x_context, y_context, x_target, y_target = random_split_context_target(
                x_train[task], y_train[task], np.random.randint(2,5))

            # type conversion
            x_c = torch.from_numpy(x_context)
            x_t = torch.from_numpy(x_target)
            y_c = torch.from_numpy(y_context)
            y_t = torch.from_numpy(y_target)

            x_ct = torch.cat([x_c, x_t], dim=0)
            y_ct = torch.cat([y_c, y_t], dim=0)

            # Get latent representation for target + context
            z_mean_all, z_std_all = data_to_z_params(x_ct, y_ct, r_encoder, z_encoder)
            z_mean_context, z_std_context = data_to_z_params(x_c, y_c,r_encoder, z_encoder)

            # sample batch of z's using repraram trick
            zs = sample_z(z_mean_all, z_std_all, n_z_samples, z_dim)
            mu, std = decoder(x_t, zs) # <- get the posterior of y*

            # Compute loss and backprop
            loss = -1 * log_likelihood(mu, std, y_t) + KLD_gaussian(
                z_mean_all, z_std_all, z_mean_context, z_std_context)

            losses.append(loss)
            loss.backward()
            opt.step()

            # store the z params
            if t == (n_epochs - 1):
                z_param_dict[task] = [data_to_z_params(x_ct, y_ct, r_encoder, z_encoder)]

    return r_encoder, z_encoder, decoder, z_param_dict

def NP_train_swap(x_train, y_train, dim_struc, n_z_samples, n_epochs):
    """
    @param: x_train, y_train - dictionaries contian stuff for different tasks
    @param: dim_struc - dimensionality structure of our networks
    @param: n_z_samples - number of z sampels
    @param: n_epochs - number of training
    """

    # setup neural net structure
    r_dim, z_dim, x_dim, y_dim = dim_struc

    # setup coders
    r_encoder = REncoder(x_dim+y_dim, r_dim)
    z_encoder = ZEncoder(r_dim, z_dim)
    decoder = Decoder(x_dim+z_dim, y_dim)

    # setup optimiser
    opt = torch.optim.Adam(list(decoder.parameters())
                        +list(z_encoder.parameters())
                        +list(r_encoder.parameters()), 1e-3)

    # setup training
    losses = []
    z_param_dict = dict()
    task_ls = ["task " + str(i) for i in range(20)]

    for t in range(n_epochs):
        for task in task_ls:
            opt.zero_grad()

            # data gen
            x_context, y_context, x_target, y_target = random_split_context_target(
                x_train[task], y_train[task], np.random.randint(2,5))

            # type conversion
            x_c = torch.from_numpy(x_context)
            x_t = torch.from_numpy(x_target)
            y_c = torch.from_numpy(y_context)
            y_t = torch.from_numpy(y_target)

            x_ct = torch.cat([x_c, x_t], dim=0)
            y_ct = torch.cat([y_c, y_t], dim=0)

            # Get latent representation for target + context
            z_mean_all, z_std_all = data_to_z_params(x_ct, y_ct, r_encoder, z_encoder)
            z_mean_context, z_std_context = data_to_z_params(x_c, y_c,r_encoder, z_encoder)

            # sample batch of z's using repraram trick
            zs = sample_z(z_mean_all, z_std_all, n_z_samples, z_dim)
            mu, std = decoder(x_t, zs) # <- get the posterior of y*

            # Compute loss and backprop
            loss = -1 * log_likelihood(mu, std, y_t) + KLD_gaussian(
                z_mean_all, z_std_all, z_mean_context, z_std_context)

            losses.append(loss)
            loss.backward()
            opt.step()

            # store the z params
            if t == (n_epochs - 1):
                z_param_dict[task] = [data_to_z_params(x_ct, y_ct, r_encoder, z_encoder)]

    return r_encoder, z_encoder, decoder, z_param_dict
