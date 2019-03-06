import numpy as np
import torch
import matplotlib.pyplot as plt

class REncoder(torch.nn.Module):
    """
    Encodes inputs of the form (x_i, y_i) into representations, r_i
    """
    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):
        super(REncoder, self).__init__()
        self.l1_size = 8
        self.l2_size = 16

        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        self.l3 = torch.nn.Linear(self.l2_size, out_dim)

        self.a = torch.nn.ReLU()
        self.b = torch.nn.ReLU()

        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)

    def forward(self, inputs):
        return self.l3(self.b(self.l2(self.a(self.l1(inputs)))))

class ZEncoder(torch.nn.Module):
    """
    Takes an r representations and produces the mean & stf of the latent
    representation z
    """
    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):

        super(ZEncoder, self).__init__()
        self.m1_size = out_dim
        self.std1_size = out_dim

        self.m1 = torch.nn.Linear(in_dim, self.m1_size)
        self.std1 = torch.nn.Linear(in_dim, self.std1_size)

        if init_func is not None:
            init_func(self.m1.weight)
            init_func(self.std1.weight)

    def forward(self, inputs):
        softplus = torch.nn.Softplus()
        return self.m1(inputs), softplus(self.std1(inputs))

class Decoder(torch.nn.Module):
    """
    Takes the x_star points, along with a function encoding z, to make
    predictions
    """

    def __init__(self, in_dim, out_dim, init_func=torch.nn.init.normal_):

        super(Decoder, self).__init__()
        self.l1_size = 8
        self.l2_size = 16

        self.l1 = torch.nn.Linear(in_dim, self.l1_size)
        self.l2 = torch.nn.Linear(self.l1_size, self.l2_size)
        self.l3 = torch.nn.Linear(self.l2_size, out_dim)

        if init_func is not None:
            init_func(self.l1.weight)
            init_func(self.l2.weight)
            init_func(self.l3.weight)

        self.a = torch.nn.Sigmoid()
        self.b = torch.nn.Sigmoid()

    def forward(self, x_pred, z):
        """
        Not sure what is happening
        """
        zs_reshaped = z.unsqueeze(-1).expand(z.shape[0], z.shape[1], x_pred.shape[0]).transpose(1,2)
        xpred_reshaped = x_pred.unsqueeze(0).expand(z.shape[0], x_pred.shape[0], x_pred.shape[1])

        xz = torch.cat([xpred_reshaped, zs_reshaped], dim=2)
        return self.l3(self.b(self.l2(self.a(self.l1(xz))))).squeeze(-1).transpose(0,1), 0.005


def log_likelihood(mu, std, target):
    norm = torch.distributions.Normal(mu, std)
    return norm.log_prob(target).sum(dim=0).mean()

def KLD_gaussian(mu_q, std_q, mu_p, std_p):
    """Analytical KLD between 2 Gaussians."""
    qs2 = std_q**2 + 1e-16
    ps2 = std_p**2 + 1e-16

    return (qs2/ps2 + ((mu_q-mu_p)**2)/ps2 + torch.log(ps2/qs2) - 1.0).sum()*0.5

def random_split_context_target(x,y, n_context):
    """Helper function to split randomly into context and target"""
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)

def sample_z(mu, std, n, z_dim):
    """Reparameterisation trick."""
    eps = torch.autograd.Variable(std.data.new(n,z_dim).normal_())
    return mu + std * eps

def data_to_z_params(x, y, r_encoder, z_encoder):
    """Helper to batch together some steps of the process."""
    xy = torch.cat([x,y], dim=1)
    rs = r_encoder(xy)
    r_agg = rs.mean(dim=0) # Average over samples
    return z_encoder(r_agg) # Get mean and variance for q(z|...)

def visualise(x, y, x_star,r_encoder, z_encoder, decoder, z_dim):
    z_mu, z_std = data_to_z_params(x,y,r_encoder, z_encoder)
    zsamples = sample_z(z_mu, z_std, 100, z_dim)

    mu, _ = decoder(x_star, zsamples)
    for i in range(mu.shape[1]):
        plt.plot(x_star.data.numpy(), mu[:,i].data.numpy(), linewidth=0.5, alpha=0.4)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.show()

def sub_visualise(x, y, x_star, r_encoder, z_encoder, decoder, z_dim, axes,title):
    z_mu, z_std = data_to_z_params(x,y,r_encoder, z_encoder)
    zsamples = sample_z(z_mu, z_std, 100, z_dim)

    mu, _ = decoder(x_star, zsamples)
    for i in range(mu.shape[1]):
        axes.plot(x_star.data.numpy(), mu[:,i].data.numpy(), linewidth=0.5, alpha=0.4)
        axes.set_title(title)
    axes.scatter(x.data.numpy(), y.data.numpy(), s=30)
