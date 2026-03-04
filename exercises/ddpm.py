# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        # neg_elbo = 0

        # First flatten input fot network, so it has shape (batch_size, D)
        x = x.flatten(1)

        # Sample time steps t ~ Uniform({1, ..., T}) implemented as {0, ..., T-1}
        t = torch.randint(0, self.T, (x.size(0), 1), device=x.device)

        # Get corresponding alpha_t values and their square roots
        alpha_bar_t = self.alpha_cumprod[t]

        # Sample noise eps ~ N(0, I)
        noise = torch.randn_like(x)

        # Sample x_t using the closed-form q(x_t | x_0)
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        # Normalised time input for the network in [0, 1]
        t_norm = (t + 1) / self.T

        # Predict the noise eps_theta(x_t, t)
        noise_pred = self.network(x_t, t_norm)

        # Negative ELBO (up to a constant) given by the per-sample MSE between eps and eps_theta
        neg_elbo = ((noise_pred - noise) ** 2).mean(dim=1)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###

            # Time step tensor for the network
            t_tensor = torch.ones(x_t.size(0), 1, device=x_t.device) * t

            # Normalised time input for the network in [0, 1]
            t_norm = (t_tensor + 1) / self.T

            # Flatten x_t for the network
            x_t_flat = x_t.flatten(1) 

            # Predict the noise eps_theta(x_t, t)
            noise_pred = self.network(x_t_flat, t_norm).view_as(x_t)

            # Get alpha_t, beta_t and alpha_bar_t for this time step
            alpha_t = self.alpha[t]
            beta_t = self.beta[t]
            alpha_bar_t = self.alpha_cumprod[t]

            # Compute the mean of p_theta(x_{t-1} | x_t)
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = coef1 * (x_t - coef2 * noise_pred)

            # Add Gaussian noise with variance beta_t, except for the last step
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean
                
        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
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
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f" {loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData
    from fid import compute_fid

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist', 'latent'], help='dataset to use {tg: two Gaussians, cb: chequerboard, mnist: MNIST, latent: VAE latent space} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples/ddpm_samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--arch', type=str, default='fc', choices=['fc', 'unet'], help='network architecture for MNIST {fc, unet} (default: %(default)s)')
    parser.add_argument('--vae-model', type=str, default='vae_model.pt', help='file to load VAE model from when using latent DDPM (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable for latent DDPM (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the data
    n_data = 10000000
    if args.data in ['tg', 'cb']:
        toy_class = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]
        toy = toy_class()
        transform = lambda x: (x-0.5)*2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
    else:
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.flatten())
        ])
        latent_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.squeeze())
        ])
        transform = latent_transform if args.data == 'latent' else mnist_transform
        train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('data/', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # Latent DDPM: override loaders to yield z ~ q(z|x) from a pre-trained VAE
    if args.data == 'latent':
        from vae_bernoulli import VAE, GaussianPrior, GaussianEncoder, GaussianDecoder  # adjust import/module name if needed

        # Build the VAE architecture exactly as used in training (assumed correct)
        M = args.latent_dim
        prior = GaussianPrior(M)

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
            nn.Linear(512, 784 * 2),  # outputs mu and log_var concatenated flat
        )

        decoder = GaussianDecoder(decoder_net)
        encoder = GaussianEncoder(encoder_net)
        vae = VAE(prior, decoder, encoder).to(args.device)
        vae.load_state_dict(torch.load(args.vae_model, map_location=torch.device(args.device)))
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)

        class LatentDataset(torch.utils.data.IterableDataset):
            def __init__(self, base_loader, vae, device):
                super().__init__()
                self.base_loader = base_loader
                self.vae = vae
                self.device = device

            def __iter__(self):
                for batch in self.base_loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x = x.to(self.device)
                    with torch.no_grad():
                        q = self.vae.encoder(x)
                        z = q.rsample()
                    yield z

            def __len__(self):
                return len(self.base_loader)

        # Reuse the already-defined MNIST loaders as base loaders
        train_loader = torch.utils.data.DataLoader(LatentDataset(train_loader, vae, args.device), batch_size=None)
        test_loader = torch.utils.data.DataLoader(LatentDataset(test_loader, vae, args.device), batch_size=None)

    # Get the dimension of the dataset
    first_batch = next(iter(train_loader))
    if isinstance(first_batch, (list, tuple)):
        first_batch = first_batch[0]
    D = first_batch.shape[1]

    # Define the network
    if args.data in ['tg', 'cb']:
        num_hidden = 64
        network = FcNetwork(D, num_hidden)
    else:
        if args.arch == 'fc' or args.data == 'latent':
            num_hidden = 64
            network = FcNetwork(D, num_hidden)
        else:
            from unet import Unet
            network = Unet()

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np
        import time

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        model.eval()

        if args.data == 'latent':
            with torch.no_grad():
                z = model.sample((4, D)).to(args.device)
                x = vae.decoder(z).sample().view(-1, 1, 28, 28).cpu()
            save_image(x.clamp(0.0, 1.0), args.samples, nrow=4)

        elif args.data == 'mnist':
            with torch.no_grad():
                samples = model.sample((10000, D)).to(args.device)
            samples = (samples / 2 + 0.5).clamp(0.0, 1.0).view(-1, 1, 28, 28)

            real_batches = []
            for real_batch in test_loader:
                if isinstance(real_batch, (list, tuple)):
                    real_batch = real_batch[0]
                real_batches.append(real_batch)
            x_real = torch.cat(real_batches, dim=0)
            x_real = (x_real / 2 + 0.5).clamp(0.0, 1.0).view(-1, 1, 28, 28).to(args.device)

            fid = compute_fid(
                x_real,
                samples,
                device=args.device,
                classifier_ckpt="models/mnist_classifier.pth",
            )
            print(f"FID (DDPM, MNIST): {fid}")
            save_image(samples[:4], args.samples, nrow=4)

        else:
            with torch.no_grad():
                samples = model.sample((10000, D)).to(args.device)
            samples = (samples / 2 + 0.5)

            toy_class = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]
            toy = toy_class()
            coordinates = [[[x, y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(
                prob,
                extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]],
                origin='lower',
                cmap='YlOrRd',
            )
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(args.samples)
            plt.close()

        # Measure samples/s using 128 batch size over 1000 iterations
        n_iters = 1000
        iter_start = time.time()
        with torch.no_grad():
            for _ in range(n_iters):
                model.sample((128, D))
        print(f"Samples per second (wallclock): {n_iters * 128 / (time.time() - iter_start):.2f}")