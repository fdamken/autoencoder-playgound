import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common import test_data
from src.mnist_autoencoder import AutoEncoder
from src.mnist_conv_autoencoder import ConvAutoEncoder
from src.mnist_variational_autoencoder import VariationalAutoEncoder


if __name__ == '__main__':
    parser = ArgumentParser('Analyzes the specified auto-encoder results. The tensorboard exports are expected to have the names "<type>-b<bottleneck>-<metric>.csv", ' +
                            'e.g. "vae-b003-test_loss.csv" and to lie in the input directory. The model file is expected to have the name "<type>-b<bottleneck>.model".')
    parser.add_argument('type', help = 'The type of the model (AE, CAE or VAE, for auto-encoder, convolutional auto-encoder or variational auto-encoder, respectively).')
    parser.add_argument('-b', '--bottleneck', type = int, help = 'Number of latent variables in the bottleneck (if required for the model).')
    parser.add_argument('-i', '--input', default = 'results/raw', help = 'The input directory. Defaults to "results/raw".')
    parser.add_argument('-o', '--output', default = 'results/figures', help = 'The output directory for all figures. Defaults to "results/figures".')
    parser.add_argument('-s', '--silent', action = 'store_false', help = 'Disable showing of the plots (but still exporting the figures).')
    args = parser.parse_args()
    model_type = args.type.lower()
    bottleneck = args.bottleneck
    input_dir = args.input
    output_dir = args.output
    show_plots = args.silent
    if bottleneck is None:
        model_file = '%s/%s.model' % (input_dir, model_type)
        tb_data_file = lambda label: '%s/%s-%s.csv' % (input_dir, model_type, label)
        figure_base_name = '%s/%s' % (output_dir, model_type)
    else:
        model_file = '%s/%s-b%03d.model' % (input_dir, model_type, bottleneck)
        tb_data_file = lambda label: '%s/%s-b%03d-%s.csv' % (input_dir, model_type, bottleneck, label)
        figure_base_name = '%s/%s-b%03d' % (output_dir, model_type, bottleneck)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_type == 'ae':
        if bottleneck is None:
            raise Exception('Model type <AE> requires a bottleneck!')
        model = AutoEncoder(bottleneck)
    elif model_type == 'cae':
        if bottleneck is not None:
            raise Exception('Model type <CAE> does not allow a bottleneck!')
        model = ConvAutoEncoder()
    elif model_type == 'vae':
        if bottleneck is None:
            raise Exception('Model type <VAE> requires a bottleneck!')
        model = VariationalAutoEncoder(bottleneck)
    else:
        raise Exception(f'Unknown model type <{model_type}>!')

    model.load_state_dict(torch.load(model_file, map_location = torch.device('cpu')))



    def save_figure(fig, comment):
        fig.savefig(figure_base_name + ('-%s.%s' % (comment, 'png')), dpi = 150)
        fig.savefig(figure_base_name + ('-%s.%s' % (comment, 'pgf')), dpi = 150)



    def plot_reconstruction_loss():
        train_reconstruction_loss = np.genfromtxt(tb_data_file('train_reconstruction_loss' if model_type == 'vae' else 'train_loss'), delimiter = ',', skip_header = 1)[:, 1:]
        test_reconstruction_loss = np.genfromtxt(tb_data_file('test_reconstruction_loss' if model_type == 'vae' else 'test_loss'), delimiter = ',', skip_header = 1)[:, 1:]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(*train_reconstruction_loss.T, label = 'Train', alpha = 0.5)
        ax.plot(*test_reconstruction_loss.T, label = 'Test')
        if bottleneck is None:
            ax.set_title('Reconstruction Loss (%s)' % model_type.upper())
        else:
            ax.set_title('Reconstruction Loss (%s, Bottleneck: %d)' % (model_type.upper(), bottleneck))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reconstruction Loss')
        ax.legend()
        save_figure(fig, 'reconstruction-loss')
        if show_plots:
            fig.show()



    def plot_kl_divergence():
        train_kl_divergence = np.genfromtxt(tb_data_file('train_kl_divergence'), delimiter = ',', skip_header = 1)[:, 1:]
        test_kl_divergence = np.genfromtxt(tb_data_file('test_kl_divergence'), delimiter = ',', skip_header = 1)[:, 1:]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(*train_kl_divergence.T, label = 'Train', alpha = 0.5)
        ax.plot(*test_kl_divergence.T, label = 'Test')
        if bottleneck is None:
            ax.set_title('KL Divergence (%s)' % model_type.upper())
        else:
            ax.set_title('KL Divergence (%s, Bottleneck: %d)' % (model_type.upper(), bottleneck))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('KL Divergence')
        ax.legend()
        save_figure(fig, 'kl-divergence')
        if show_plots:
            fig.show()



    def plot_elbo():
        train_elbo = np.genfromtxt(tb_data_file('train_loss'), delimiter = ',', skip_header = 1)[:, 1:] * np.array([1, -1])
        test_elbo = np.genfromtxt(tb_data_file('test_loss'), delimiter = ',', skip_header = 1)[:, 1:] * np.array([1, -1])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(*train_elbo.T, label = 'Train', alpha = 0.5)
        ax.plot(*test_elbo.T, label = 'Test')
        if bottleneck is None:
            ax.set_title('Evidence Lower Bound (%s)' % model_type.upper())
        else:
            ax.set_title('Evidence Lower Bound (%s, Bottleneck: %d)' % (model_type.upper(), bottleneck))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Evidence Lower Bound')
        ax.legend()
        save_figure(fig, 'elbo')
        if show_plots:
            fig.show()



    def plot_latent_space():
        latent_values = { }
        for batch, labels in test_data:
            encoded = model.encode(batch.view(batch.size(0), -1))
            latents = encoded[0] if model_type == 'vae' else encoded
            for i, (label, latent) in enumerate(zip(labels, latents.tolist())):
                label = str(label.item())
                if label not in latent_values:
                    latent_values[label] = []
                latent_values[label].append(latent)

        fig = plt.figure()
        if bottleneck >= 3:
            ax = fig.add_subplot(1, 1, 1, projection = '3d')
        else:
            ax = fig.add_subplot(1, 1, 1)
        for label, latents in sorted(latent_values.items()):
            ax.scatter(*map(list, zip(*latents)), label = label, s = 2, alpha = 0.5)
        ax.legend()
        save_figure(fig, 'latent-space')
        if show_plots:
            fig.show()



    # Invoke all the methods.
    plot_reconstruction_loss()
    if model_type == 'vae':
        plot_kl_divergence()
    if model_type == 'vae':
        plot_elbo()
    if bottleneck is not None and bottleneck <= 3:
        # Plot the latent space for AE and VAE (if trivially visualizable, i.e. dim <= 3)
        plot_latent_space()
