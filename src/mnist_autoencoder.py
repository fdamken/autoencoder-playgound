import argparse
import os
import shutil

import torch.utils.data
import torchvision
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter

from src.common import test_data, train_data


torch.manual_seed(42)

LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
WRITE_IMAGE_EVERY_N_EPOCHS = 10
NAME = os.path.basename(__file__).replace('.py', '')



class AutoEncoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(AutoEncoder, self).__init__()

        self._encoder = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(True),
                nn.Linear(128, 64),
                nn.ReLU(True),
                nn.Linear(64, 12),
                nn.ReLU(True),
                nn.Linear(12, bottleneck_size),
                nn.ReLU(True)
        )
        self._decoder = nn.Sequential(
                nn.Linear(bottleneck_size, 12),
                nn.ReLU(True),
                nn.Linear(12, 64),
                nn.ReLU(True),
                nn.Linear(64, 128),
                nn.ReLU(True),
                nn.Linear(128, 784),
                nn.Tanh()
        )


    def encode(self, x):
        return self._encoder(x)


    def decode(self, latent):
        return self._decoder(latent)


    def forward(self, x):
        return self.decode(self.encode(x))



def loss_fn(img: torch.Tensor, reconstruction: torch.Tensor):
    return ((img - reconstruction) ** 2).sum(1).mean()



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

    ae = AutoEncoder(bottleneck_size).to(device)
    optimizer = optim.Adam(ae.parameters(), lr = LEARNING_RATE, weight_decay = 1e-5)
    writer = SummaryWriter(comment = tb_comment)
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = 0
        for img, _ in train_data:
            img = img.view(img.size(0), -1)
            img = img.to(device)
            out = ae(img)

            optimizer.zero_grad()
            loss = loss_fn(out, img)
            train_loss += loss
            loss.backward()
            optimizer.step()
        train_loss /= len(train_data)

        with torch.no_grad():
            test_loss = 0
            for img, _ in test_data:
                img = img.view(img.size(0), -1)
                img = img.to(device)
                out = ae(img)

                test_loss += loss_fn(out, img)
            test_loss /= len(test_data)

        print('Epoch %5d: train_loss=%.5f, test_loss=%.5f' % (epoch, train_loss, test_loss))
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)

        if epoch % WRITE_IMAGE_EVERY_N_EPOCHS == 0:
            # noinspection PyUnboundLocalVariable
            writer.add_image('original', torchvision.utils.make_grid(img.view(out.size(0), 1, 28, 28)), epoch)
            # noinspection PyUnboundLocalVariable
            writer.add_image('reconstruction', torchvision.utils.make_grid(out.view(out.size(0), 1, 28, 28)), epoch)
            torchvision.utils.save_image(img.view(img.size(0), 1, 28, 28), f'{img_out_directory}/img_%05d.png' % epoch)
            torchvision.utils.save_image(out.view(out.size(0), 1, 28, 28), f'{img_out_directory}/out_%05d.png' % epoch)
    writer.close()

    torch.save(ae.state_dict(), f'{NAME}-bottleneck={bottleneck_size}.model')
