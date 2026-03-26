# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

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


class GaussianDecoders(nn.Module):
    def __init__(self, decoder_nets):
        """
        Define a list of Bernoulli decoder distributions based on given decoder networks.

        Parameters:
        encoder_nets: [list[torch.nn.Module]]
           The decoder networks that take as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoders, self).__init__()
        self.decoder_nets = nn.ModuleList(decoder_nets)
        self.active_idx: int = 0
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def active_decoder(self, idx):
        """Set the active decoder. Relevant during training."""
        self.active_idx=idx

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space
        from the active decoder network.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_nets[self.active_idx](z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)

    def all_means(self, z):
        """Return the means of all decoders for a batch of latent variables.
        
        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        return torch.stack([net(z) for net in self.decoder_nets])


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior: GaussianPrior, decoders: GaussianDecoders, encoder: GaussianEncoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoders: [GaussianDecoders]
              The decoder distributions over the data space.
        encoder: [GaussianEncoder]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoders = decoders
        self.encoder = encoder

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

        elbo = torch.mean(
            self.decoders(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoders(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model: VAE, optimizer, data_loader, epochs, device):
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
    num_decoders = len(model.decoders.decoder_nets)
    num_steps = len(data_loader) * num_decoders * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            # Rotate active decoder every step
            active_idx = epoch % len(model.decoders.decoder_nets)
            model.decoders.active_decoder(active_idx)
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch+1}, decoder={active_idx+1} step={step}, loss={loss:.1f}"
                    )   

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch+1} and current loss: {loss:.1f}"
                )
                break


def decoder_mean(decoder, z):
    """
    Return the mean of the Gaussian decoder for latent variables `z`.
    """
    return decoder(z).mean


def linear_paths(starts, ends, num_points):
    """
    Create piecewise-linear paths between batches of start/end latent points.
    """
    t = torch.linspace(0.0, 1.0, num_points, device=starts.device, dtype=starts.dtype)
    return (1.0 - t[None, :, None]) * starts[:, None, :] + t[None, :, None] * ends[:, None, :]


def curve_energy(decoder, curves):
    """
    Approximate the pull-back energy of curves using finite differences in data space.

    Parameters:
    decoder: [torch.nn.Module]
       Decoder distribution p(x|z).
    curves: [torch.Tensor]
       Tensor of shape `(batch_size, num_points, latent_dim)`.

    Returns:
    energies: [torch.Tensor]
       Tensor of shape `(batch_size,)` with one energy per curve.
    """
    batch_size, num_points, latent_dim = curves.shape
    decoded = decoder_mean(decoder, curves.reshape(batch_size * num_points, latent_dim))
    decoded = decoded.reshape(batch_size, num_points, *decoded.shape[1:])
    diffs = decoded[:, 1:] - decoded[:, :-1]
    dt = 1.0 / max(num_points - 1, 1)
    return diffs.flatten(start_dim=2).pow(2).sum(dim=(1, 2)) / dt


def curve_energy_ensemble(decoder: GaussianDecoders, curves, exact=False):
    batch_size, num_points, latent_dim = curves.shape
    flat_curves = curves.reshape(batch_size * num_points, latent_dim)

    # Shape: (M, batch_size * num_points, ...)
    all_decoded = decoder.all_means(flat_curves)
    M_decoders = all_decoded.shape[0]
    
    # Reshape to easily index time steps
    all_decoded = all_decoded.reshape(M_decoders, batch_size, num_points, -1)
    dt = 1.0 / max(num_points - 1, 1)

    if exact:
        decode_i = all_decoded[:, :, :-1].unsqueeze(1)
        decode_j = all_decoded[:, :, 1:].unsqueeze(0)
        diffs = decode_i - decode_j
        energies_per_pair = diffs.pow(2).sum(dim=(3, 4)) / dt
        return energies_per_pair.mean(dim=(0, 1))
    else:
        # Monte Carlo approximation for the optimization loop
        i = torch.randint(M_decoders, (1,)).item()
        j = torch.randint(M_decoders, (1,)).item()
        
        diffs = all_decoded[i, :, :-1] - all_decoded[j, :, 1:]
        return diffs.pow(2).sum(dim=(1, 2)) / dt


def compute_geodesics_ensemble(decoder, starts, ends, num_points, num_steps, lr):
    """
    Optimize interior points of piecewise-linear curves to minimize the expected 
    pull-back energy across an ensemble of decoders.
    """
    curves = linear_paths(starts, ends, num_points)
    
    if num_points <= 2:
        # If the curve only has a start and end point, return the exact expected energy
        return curves, curve_energy_ensemble(decoder, curves, exact=True)

    interior = nn.Parameter(curves[:, 1:-1].clone())
    optimizer = torch.optim.Adam([interior], lr=lr)

    with torch.no_grad():
        best_curves = curves.clone()
        # Evaluate the baseline using the EXACT expected energy
        best_energies = curve_energy_ensemble(decoder, best_curves, exact=True)

    for _ in range(num_steps):
        optimizer.zero_grad()
        # Reconstruct the full curve (start + interior + end)
        curves = torch.cat([starts[:, None, :], interior, ends[:, None, :]], dim=1)
        
        mc_energies = curve_energy_ensemble(decoder, curves, exact=False)
        loss = mc_energies.mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            exact_energies = curve_energy_ensemble(decoder, curves, exact=True)
            improved = exact_energies < best_energies
            best_energies[improved] = exact_energies[improved]
            best_curves[improved] = curves.detach()[improved]

    return best_curves, best_energies


def compute_geodesics(decoder, starts, ends, num_points, num_steps, lr):
    """
    Optimize interior points of piecewise-linear curves to minimize pull-back energy.
    """
    curves = linear_paths(starts, ends, num_points)
    if num_points <= 2:
        return curves, curve_energy(decoder, curves)

    interior = nn.Parameter(curves[:, 1:-1].clone())
    optimizer = torch.optim.Adam([interior], lr=lr)

    with torch.no_grad():
        best_curves = curves.clone()
        best_energies = curve_energy(decoder, best_curves)

    for _ in range(num_steps):
        optimizer.zero_grad()
        curves = torch.cat([starts[:, None, :], interior, ends[:, None, :]], dim=1)
        energies = curve_energy(decoder, curves)
        loss = energies.mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            improved = energies < best_energies
            best_energies[improved] = energies[improved]
            best_curves[improved] = curves.detach()[improved]

    return best_curves, best_energies


def collect_latents(model, data_loader, device):
    """
    Encode a dataset into posterior means in latent space.
    """
    latents = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            latents.append(model.encoder(x).mean.cpu())
            labels.append(y.cpu())
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)


def sample_latent_pairs(latents, num_pairs, seed):
    """
    Sample disjoint random pairs of latent points.
    """
    if latents.size(0) < 2:
        raise ValueError("Need at least two latent points to compute geodesics.")

    max_pairs = latents.size(0) // 2
    num_pairs = min(num_pairs, max_pairs)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(latents.size(0), generator=generator)
    pair_idx = perm[: 2 * num_pairs].reshape(num_pairs, 2)
    return pair_idx


def plot_latent_geodesics(latents, labels, pair_idx, curves, output_file):
    """
    Plot latent embeddings and optimized geodesics.
    """
    plt.figure(figsize=(8.5, 8.5))
    for label in labels.unique(sorted=True):
        mask = labels == label
        plt.scatter(
            latents[mask, 0],
            latents[mask, 1],
            s=18,
            alpha=0.55,
            label=f"class {int(label)}",
        )

    for curve_id, curve in enumerate(curves):
        curve = curve.cpu()
        plt.plot(curve[:, 0], curve[:, 1], color="black", alpha=0.38, linewidth=1.2)
        start_idx, end_idx = pair_idx[curve_id].tolist()
        endpoints = latents[[start_idx, end_idx]]
        plt.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            color="black",
            s=28,
            alpha=0.7,
        )

    plt.xlabel("z1", fontsize=16)
    plt.ylabel("z2", fontsize=16)
    plt.title("Latent Space with Pull-back Geodesics", fontsize=22, pad=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="checkpoint file to save/load the model from (default: <experiment-folder>/model.pt)",
    )
    parser.add_argument(
        "--geodesics-file",
        type=str,
        default=None,
        help="file to save the geodesics plot in (default: <experiment-folder>/geodesics.png)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=25,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )
    parser.add_argument(
        "--geodesic-iters",
        type=int,
        default=300,
        metavar="N",
        help="number of optimization steps used for geodesics (default: %(default)s)",
    )
    parser.add_argument(
        "--geodesic-lr",
        type=float,
        default=1e-2,
        metavar="LR",
        help="learning rate used for geodesic optimization (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="random seed used when sampling point pairs (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    torch.manual_seed(args.seed)
    device = args.device
    model_file = args.model_file or os.path.join(args.experiment_folder, "model.pt")
    geodesics_file = args.geodesics_file or os.path.join(args.experiment_folder, "geodesics.png")

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":
        for rerun in range(args.num_reruns):
            experiments_folder = args.experiment_folder
            model_file = os.path.join(args.experiment_folder, f"multidecode_rerun_{rerun+1}_D_{args.num_decoders}.pt")

            os.makedirs(f"{experiments_folder}", exist_ok=True)
            os.makedirs(os.path.dirname(model_file) or ".", exist_ok=True)
            
            model = VAE(
                GaussianPrior(M),
                GaussianDecoders([new_decoder() for _ in range(args.num_decoders)]),
                GaussianEncoder(new_encoder()),
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(
                model,
                optimizer,
                mnist_train_loader,
                args.epochs_per_decoder,
                args.device,
            )
            os.makedirs(f"{experiments_folder}", exist_ok=True)
            torch.save(model.state_dict(), model_file)

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoders([new_decoder() for _ in range(args.num_decoders)]),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(model_file))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        for rerun in range(args.num_reruns):
            model_file = os.path.join(args.experiment_folder, f"multidecode_rerun_{rerun+1}_D_{args.num_decoders}.pt")
            model = VAE(
                GaussianPrior(M),
                GaussianDecoders([new_decoder() for _ in range(args.num_decoders)]),
                GaussianEncoder(new_encoder()),
            ).to(device)
            model.load_state_dict(torch.load(model_file))
            model.eval()

            elbos = []
            with torch.no_grad():
                for x, y in mnist_test_loader:
                    x = x.to(device)
                    elbo = model.elbo(x)
                    elbos.append(elbo)
            mean_elbo = torch.tensor(elbos).mean()
            print(f"Mean test ELBO [rerun {rerun+1}]:", mean_elbo)

    elif args.mode == "geodesics":

        # Ensure same testing points across the models
        images, labels = next(iter(mnist_test_loader))
        images = images.to(device)

        num_pairs = 10

        if images.shape[0] < num_pairs * 2:
            raise ValueError(f"Expected a batch size equal to {num_pairs*2} at least")

        starts_img = images[:num_pairs]
        ends_img = images[num_pairs:2*num_pairs]

        cov_results = {'euclidean': [], 'geodesic': []}

        for num_dec in [1,2,3]:

            euc_dists_for_D = []
            geo_dists_for_D = []

            for rerun in range(args.num_reruns):
                model_file = os.path.join(args.experiment_folder, f"multidecode_rerun_{rerun+1}_D_{num_dec}.pt")
                model = VAE(
                    GaussianPrior(M),
                    GaussianDecoders([new_decoder() for _ in range(num_dec)]),
                    GaussianEncoder(new_encoder()),
                ).to(device)
                model.load_state_dict(torch.load(model_file))
                model.eval()

                with torch.no_grad():
                    starts_z = model.encoder(starts_img).mean
                    ends_z = model.encoder(ends_img).mean
                
                # A. Euclidean Distance
                euc_dist = torch.norm(starts_z - ends_z, dim=-1)
                euc_dists_for_D.append(euc_dist)
                
                # B. Geodesic Distance
                curves, energies = compute_geodesics_ensemble(
                    model.decoders, 
                    starts_z,
                    ends_z,
                    num_points=args.num_t, 
                    num_steps=args.geodesic_iters, 
                    lr=args.geodesic_lr
                )
                
                # Because the curves are optimized to have constant speed, length = sqrt(Energy). We use this as our distance measure.
                geo_dist = torch.sqrt(energies.detach())
                geo_dists_for_D.append(geo_dist)

            # Stack lists to shape: (num_reruns, num_pairs)
            euc_stack = torch.stack(euc_dists_for_D)
            geo_stack = torch.stack(geo_dists_for_D)
            
            # CoV = standard deviation / mean (across the model dimension 0).
            euc_cov = euc_stack.std(dim=0) / euc_stack.mean(dim=0)
            geo_cov = geo_stack.std(dim=0) / geo_stack.mean(dim=0)
            
            # Store the average CoV across the 10 point pairs[cite: 59].
            avg_euc_cov = euc_cov.mean().item()
            avg_geo_cov = geo_cov.mean().item()
            
            cov_results['euclidean'].append(avg_euc_cov)
            cov_results['geodesic'].append(avg_geo_cov)
        
        print("Average CoV for Euclidean Distances across reruns:", cov_results['euclidean'])
        print("Average CoV for Geodesic Distances across reruns:", cov_results['geodesic'])

        # 4. PLOT THE RESULTS
        plt.figure(figsize=(8, 6))
        M_values = [1, 2, 3]
        
        plt.plot(M_values, cov_results['euclidean'], marker='o', label='Euclidean Distance', linestyle='--', color='red')
        plt.plot(M_values, cov_results['geodesic'], marker='s', label='Geodesic Distance', linewidth=2, color='blue')
        
        plt.xlabel('Number of Ensemble Decoders (M)', fontsize=14)
        plt.ylabel('Average Coefficient of Variation (CoV)', fontsize=14)
        plt.title('Reliability of Distances vs. Ensemble Size', fontsize=16)
        plt.xticks(M_values)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        os.makedirs(args.experiment_folder, exist_ok=True)
        plot_path = os.path.join(args.experiment_folder, "cov_plot.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved final CoV plot to {plot_path}")