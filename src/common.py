import os

import torch.utils.data
from torchvision import datasets
from torchvision.transforms import transforms


BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

DATA_DIR = '../data' if os.path.basename(os.getcwd()) == 'src' else 'data'

train_data = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR,
                                                        train = True,
                                                        download = True,
                                                        transform = transforms.Compose([
                                                                transforms.ToTensor()
                                                        ])),
                                         batch_size = BATCH_SIZE,
                                         shuffle = True)
test_data = torch.utils.data.DataLoader(datasets.MNIST(DATA_DIR,
                                                       train = False,
                                                       download = False,
                                                       transform = transforms.Compose([
                                                               transforms.ToTensor()
                                                       ])),
                                        batch_size = TEST_BATCH_SIZE,
                                        shuffle = True)
