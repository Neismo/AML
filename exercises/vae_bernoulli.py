# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

from flow import MaskedCouplingLayer
from flow import Flow
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm


def plot_prior_vs_posterior(model, test_loader, device, save_path):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import torch
    from tqdm import tqdm

    model.eval()
    latents = []
    labels = []

    # 1. Collect Aggregated Posterior Samples
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Encoding Posterior"):
            x = x.to(device)
            # Assuming model.encoder returns a distribution object
            q = model.encoder(x)
            z_post = q.mean.cpu() 
            latents.append(z_post)
            labels.append(y)
    
    posterior_z = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # 2. Collect Prior Samples
    with torch.no_grad():
        # Handle different prior implementations (Flow vs Standard Normal)
        if isinstance(model.prior, Flow):
            prior_z = model.prior.sample(sample_shape=(posterior_z.shape[0],))
        else:
            prior_z = model.prior().sample(torch.Size([posterior_z.shape[0]]))
            
    prior_z = prior_z.cpu().numpy()

    # 3. Dimensionality Reduction (PCA)
    # We fit PCA on the concatenated data to ensure both are in the same 2D space
    if posterior_z.shape[1] > 2:
        pca = PCA(n_components=2)
        combined = np.vstack([posterior_z, prior_z])
        combined_2d = pca.fit_transform(combined)
        
        posterior_2d = combined_2d[:len(posterior_z)]
        prior_2d = combined_2d[len(posterior_z):]
    else:
        posterior_2d = posterior_z
        prior_2d = prior_z

    # 4. Plotting the Overlap
    plt.figure(figsize=(3, 3))
    
    # Plot Prior first (background)
    plt.scatter(prior_2d[:, 0], prior_2d[:, 1], 
                alpha=0.3, s=3, label='Prior $p(z)$', c='gray')
    
    # Plot Posterior second (foreground)
    plt.scatter(posterior_2d[:, 0], posterior_2d[:, 1], 
                alpha=0.5, s=3, label='Agg. Post. $q(z)$', c='crimson')

    plt.title(f"{type(model.prior).__name__}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=5) # Larger icons in legend for visibility
    plt.axis('equal')
    plt.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path)


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MixtureGaussianPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Mixture of Gaussians (MoG) prior with K components.

        Parameters:
        M: [int]
           Dimension of the latent space.
        K: [int]
           Number of mixture components.
        """
        super(MixtureGaussianPrior, self).__init__()
        self.M = M
        self.K = K
        self.logits = nn.Parameter(torch.zeros(self.K))
        self.loc = nn.Parameter(torch.zeros(self.K, self.M))
        self.scale = nn.Parameter(torch.ones(self.K, self.M))

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mix = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(loc=self.loc, scale=F.softplus(self.scale)), 1)
        return td.MixtureSameFamily(mix, comp)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(
            self, 
            prior: GaussianPrior | MixtureGaussianPrior | Flow, 
            decoder: BernoulliDecoder,
            encoder: GaussianEncoder,
            beta: float = 1.0,
        ):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        recon = self.decoder(z).log_prob(x)

        if isinstance(self.prior, Flow):  # flow
            log_p_z = self.prior.log_prob(z)
            log_q_z = q.log_prob(z)
            elbo = torch.mean(recon + self.beta*(log_p_z - log_q_z), dim=0)
        else:  # gauss or MoG
            prior_dist = self.prior()
            try: # Gaussian
                kl = td.kl_divergence(q, prior_dist)
            except NotImplementedError: # MoG
                kl = q.log_prob(z) - prior_dist.log_prob(z)

            elbo = torch.mean(recon - self.beta * kl, dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        if isinstance(self.prior, Flow):
            z = self.prior.sample(sample_shape=(n_samples,))
        else:
            z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'evaluate', 'plot_pca', 'plot_prior', 'plot', 'train_then_eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='prior type (default: %(default)s)')
    parser.add_argument('--mog-components', type=int, default=10, metavar='K', help='number of MoG components (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1.0, help='beta parameter for ELBO (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    if args.prior == 'mog':
        prior = MixtureGaussianPrior(M, args.mog_components)
    elif args.prior == 'flow':
        transformations=[]
        mask = torch.arange(M) % 2
        for i in range(4):
            mask = (1-mask) # Flip the mask
            scale_net = nn.Sequential(nn.Linear(M, 8), nn.ReLU(), nn.Linear(8, M), nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(M, 8), nn.ReLU(), nn.Linear(8, M))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        base = GaussianPrior(M)
        prior = Flow(base, transformations)
    else:
        prior = GaussianPrior(M)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder, beta=args.beta).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == "evaluate":
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        elbo = 0
        with torch.no_grad():
            for x in tqdm(mnist_test_loader, desc="Evaluating"):
                x = x[0].to(device)
                elbo += model.elbo(x).item()
        
        elbo /= len(mnist_test_loader)
        print(f"ELBO: {elbo:.4f}")
    
    elif args.mode == "train_then_eval":
        ELBOS = []
        for i in range(5):
            # Define prior distribution
            M = args.latent_dim
            if args.prior == 'mog':
                prior = MixtureGaussianPrior(M, args.mog_components)
            elif args.prior == 'flow':
                transformations=[]
                mask = torch.arange(M) % 2
                for i in range(4):
                    mask = (1-mask) # Flip the mask
                    scale_net = nn.Sequential(nn.Linear(M, 8), nn.ReLU(), nn.Linear(8, M), nn.Tanh())
                    translation_net = nn.Sequential(nn.Linear(M, 8), nn.ReLU(), nn.Linear(8, M))
                    transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
                base = GaussianPrior(M)
                prior = Flow(base, transformations)
            else:
                prior = GaussianPrior(M)

            # Define encoder and decoder networks
            encoder_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, M*2),
            )

            decoder_net = nn.Sequential(
                nn.Linear(M, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
                nn.Unflatten(-1, (28, 28))
            )

            # Define VAE model
            decoder = BernoulliDecoder(decoder_net)
            encoder = GaussianEncoder(encoder_net)
            model = VAE(prior, decoder, encoder).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Train model
            train(model, optimizer, mnist_train_loader, args.epochs, args.device)
            model.eval()
            elbo = 0
            with torch.no_grad():
                for x in tqdm(mnist_test_loader, desc="Evaluating"):
                    x = x[0].to(device)
                    elbo += model.elbo(x).item()
            
            elbo /= len(mnist_test_loader)
            print(f"ELBO for iteration {i}: {elbo:.4f}")
            ELBOS.append(elbo)

        print(f"Mean ELBO: {np.mean(ELBOS):.4f} ± {np.std(ELBOS):.4f}, raw ELBOs: {ELBOS}")

    elif args.mode == "plot_prior":
        import matplotlib.pyplot as plt
        model.eval()
        with torch.no_grad():
            # 1. Sample from prior
            if isinstance(model.prior, Flow):
                z = model.prior.sample(sample_shape=(5000,))
            else:
                z = model.prior().sample(torch.Size([5000]))
            z = z.cpu().numpy()

        # 2. Project to 2D if M > 2
        if z.shape[1] > 2:
            from sklearn.decomposition import PCA
            z = PCA(n_components=2).fit_transform(z)

        # 3. Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(z[:, 0], z[:, 1], alpha=0.5, s=2, c='royalblue')
        plt.title(f"Prior Distribution Latent Space ({type(model.prior).__name__})")
        plt.axis('equal')
        plt.savefig(f"exercises/samples/{type(model.prior).__name__}_prior.png")

    elif args.mode == "plot_pca":
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        # Get latent representations of test set
        latents = []
        labels = []
        with torch.no_grad():
            for x, y in tqdm(mnist_test_loader, desc="Getting latent representations"):
                x = x.to(device)
                q = model.encoder(x)
                z = q.mean.cpu()
                latents.append(z)
                labels.append(y)
        latents = torch.cat(latents, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        # Perform PCA
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        # Plot latent representations
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=8)
        plt.title("PCA of VAE Latent Space")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid()
        plt.colorbar(scatter, ticks=range(10), label="Label")
        plt.savefig(f"exercises/samples/{type(model.prior).__name__}_pca.png")

    elif args.mode == "plot":
        plot_prior_vs_posterior(model, mnist_test_loader, device, save_path=f"exercises/samples/{type(model.prior).__name__}_prior_vs_posterior.png")
