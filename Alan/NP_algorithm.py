from NP_demo import *

def NP_algo(all_x_np, all_y_np, dim_struc, n_z_samples,n_epochs, n_display, return_loss, sub_vis, axes):

    # setup dim strcutre
    r_dim, z_dim, x_dim, y_dim = dim_struc

    # setup objects
    r_encoder = REncoder(x_dim+y_dim, r_dim)
    z_encoder = ZEncoder(r_dim, z_dim)
    decoder = Decoder(x_dim+z_dim, y_dim)

    # setup optimiser
    opt = torch.optim.Adam(list(decoder.parameters())
                        +list(z_encoder.parameters())
                        +list(r_encoder.parameters()), 1e-3)

    # setup training
    losses = []
    counter = 0

    for t in range(n_epochs):
        opt.zero_grad()

        # data gen
        x_context, y_context, x_target, y_target = random_split_context_target(
            all_x_np, all_y_np, np.random.randint(1,4))

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

        if sub_vis == False:
            if t % n_display == 0:
                print(f"Function samples after {t} steps:")
                x_g = torch.from_numpy(np.arange(-4,4, 0.1).reshape(-1,1).astype(np.float32))
                visualise(x_ct, y_ct, x_g, r_encoder, z_encoder, decoder, z_dim)
        else:
            if t % n_display == 0:
                title = str(t) + "steps"
                x_g = torch.from_numpy(np.arange(-4,4, 0.1).reshape(-1,1).astype(np.float32))
                sub_visualise(x_ct, y_ct, x_g, r_encoder, z_encoder, decoder, z_dim, axes[counter], title=title)
                counter += 1

    if return_loss:
        return losses
