import argparse
import os
import shutil

import torch.utils.data
import torchvision
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms


torch.manual_seed(42)

BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
WRITE_IMAGE_EVERY_N_EPOCHS = 10
NAME = os.path.basename(__file__).replace('.py', '')



class VariationalAutoEncoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(VariationalAutoEncoder, self).__init__()

        self._fc_encoder = nn.Sequential(
                nn.Linear(784, 400),
                nn.ReLU(True)
        )
        self._fc_mean = nn.Linear(400, bottleneck_size)
        self._fc_logvar = nn.Linear(400, bottleneck_size)

        self._fc_Decoder = nn.Sequential(
                nn.Linear(bottleneck_size, 400),
                nn.ReLU(True),
                nn.Linear(400, 784),
                nn.Sigmoid()
        )


    def encode(self, x):
        fc_out = self._fc_encoder(x)
        return self._fc_mean(fc_out), self._fc_logvar(fc_out)


    def decode(self, latent):
        return self._fc_Decoder(latent)


    def forward(self, x):
        mean, logvar = self.encode(x)
        return self.decode(self._reparameterize(mean, logvar)), mean, logvar


    def _reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor):
        # Let \( \sigma' \coloneqq \text{logvar} \), then: \( \sigma' = \ln \sigma^2 \quad\iff\quad \sigma = e^{\sigma' / 2} \).
        epsilon = torch.zeros(mean.shape, requires_grad = False).to(mean.device).normal_(mean = 0, std = 1)
        std = torch.exp(logvar / 2.0)
        return mean + std * epsilon



def loss_fn(img: torch.Tensor, reconstruction: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor):
    kl = -(1 + logvar - mean ** 2 - torch.exp(logvar)).sum(1).mean() / 2.0
    reconstruction_loss = ((reconstruction - img) ** 2).sum(1).mean()
    elbo = -kl - reconstruction_loss
    return -elbo, reconstruction_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bottleneck', type = int, help = 'Number of latent variables in the bottleneck.')
    parser.add_argument('-c', '--cuda', action = 'store_true', help = 'Enable CUDA acceleration.')
    parser.add_argument('-f', '--overwrite', action = 'store_true', help = f'Overwrite image directory directory if it exists.')
    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise Exception('CUDA acceleration requested but is not available!')
    device = torch.device('cuda' if args.cuda else 'cpu')
    bottleneck_size = args.bottleneck
    img_out_directory = f'tmp_{NAME}_img-bottleneck={bottleneck_size}'
    tb_comment = f'{NAME}-bottleneck={bottleneck_size}'

    if os.path.exists(img_out_directory):
        if args.overwrite:
            shutil.rmtree(img_out_directory)
        else:
            raise Exception('Image directory %s exists!' % img_out_directory)
    os.makedirs(img_out_directory)

    train_data = torch.utils.data.DataLoader(datasets.MNIST('../data',
                                                            train = True,
                                                            download = True,
                                                            transform = transforms.Compose([
                                                                    transforms.ToTensor()
                                                            ])),
                                             batch_size = BATCH_SIZE,
                                             shuffle = True)
    test_data = torch.utils.data.DataLoader(datasets.MNIST('../data',
                                                           train = False,
                                                           download = False,
                                                           transform = transforms.Compose([
                                                                   transforms.ToTensor()
                                                           ])),
                                            batch_size = TEST_BATCH_SIZE,
                                            shuffle = True)

    vae = VariationalAutoEncoder(bottleneck_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr = LEARNING_RATE)
    writer = SummaryWriter(comment = tb_comment)
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = 0
        train_reconstruction_loss = 0
        for img, _ in train_data:
            img = img.view(img.size(0), -1)
            img = img.to(device)
            out, out_mean, out_logvar = vae(img)

            optimizer.zero_grad()
            loss, reconstruction_loss = loss_fn(img, out, out_mean, out_logvar)
            train_loss += loss
            train_reconstruction_loss += reconstruction_loss
            loss.backward()
            optimizer.step()
        train_loss /= len(train_data)
        train_reconstruction_loss /= len(train_data)

        test_loss = 0
        test_kl_divergence = 0
        test_reconstruction_loss = 0
        with torch.no_grad():
            for img, _ in test_data:
                img = img.view(img.size(0), -1)
                img = img.to(device)
                out, out_mean, out_logvar = vae(img)

                loss, reconstruction_loss = loss_fn(img, out, out_mean, out_logvar)
                test_loss += loss
                test_reconstruction_loss += reconstruction_loss
            test_loss /= len(test_data)

        print('Epoch %5d: train_loss=%.5f, test_loss=%.5f' % (epoch, train_loss, test_loss))
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_reconstruction_loss', train_reconstruction_loss, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_reconstruction_loss', test_reconstruction_loss, epoch)

        if epoch % WRITE_IMAGE_EVERY_N_EPOCHS == 0:
            # noinspection PyUnboundLocalVariable
            writer.add_image('original', torchvision.utils.make_grid(img.view(out.size(0), 1, 28, 28)), epoch)
            # noinspection PyUnboundLocalVariable
            writer.add_image('reconstruction', torchvision.utils.make_grid(out.view(out.size(0), 1, 28, 28)), epoch)
            torchvision.utils.save_image(img.view(img.size(0), 1, 28, 28), f'{img_out_directory}/img_%05d.png' % epoch)
            torchvision.utils.save_image(out.view(out.size(0), 1, 28, 28), f'{img_out_directory}/out_%05d.png' % epoch)
    writer.close()

    torch.save(vae.state_dict(), f'{NAME}-bottleneck={bottleneck_size}.model')
