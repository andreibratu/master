from typing import List
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt


class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=30):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Conv1d(1, self.hparams["num_filters"], kernel_size=3, padding=1),
            nn.MaxPool1d(2, return_indices=True),
            nn.Conv1d(self.hparams["num_filters"], 2 * self.hparams["num_filters"], kernel_size=3, padding=1),
            nn.MaxPool1d(2, return_indices=True),
            nn.Conv1d(2 * self.hparams["num_filters"], 4 * self.hparams["num_filters"], kernel_size=3, padding=1),
            nn.MaxPool1d(2, return_indices=True),
            nn.Flatten(),
            nn.Linear(1960, self.hparams['n_hidden']),
            nn.LeakyReLU(),
            nn.Dropout(p=hparams["dropout_p"]),
            nn.Linear(self.hparams['n_hidden'], self.hparams['n_hidden']),
            nn.LeakyReLU(),
            nn.Dropout(p=hparams["dropout_p"]),
            nn.Linear(self.hparams['n_hidden'], self.latent_dim),
        )

    def forward(self, x):
        # To reverse the pool operation we need to persist indices
        indices_max_pool = []
        # RUn through layers
        x = x.reshape(-1, 1, self.input_size)
        for layer in self.encoder.children():
            if isinstance(layer, nn.MaxPool1d):
                x, indices = layer.forward(x)
                indices_max_pool.append(indices)
            else:
                x = layer.forward(x)     
        return x, indices_max_pool


class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=30, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hparams['n_hidden']),
            nn.LeakyReLU(),
            nn.Dropout(p=hparams["dropout_p"]),
            nn.Linear(self.hparams['n_hidden'], self.hparams['n_hidden']),
            nn.LeakyReLU(),
            nn.Dropout(p=hparams["dropout_p"]),
            nn.Linear(self.hparams['n_hidden'], 1960),
            nn.Unflatten(1, (4 * self.hparams["num_filters"], 1960 // (4 * self.hparams["num_filters"]))),
            nn.MaxUnpool1d(2),
            nn.ConvTranspose1d(4 * self.hparams["num_filters"], 2 * self.hparams["num_filters"], kernel_size=3, padding=1),
            nn.MaxUnpool1d(2),
            nn.ConvTranspose1d(2 * self.hparams["num_filters"], self.hparams["num_filters"], kernel_size=3, padding=1),
            nn.MaxUnpool1d(2),
            nn.ConvTranspose1d(self.hparams["num_filters"], 1, kernel_size=3, padding=1),
        )

    def forward(self, x, reconstruct_indices):
        for layer in self.decoder.children():
            if isinstance(layer, nn.MaxUnpool1d):
                x = layer.forward(x, reconstruct_indices.pop())
            else:
                x = layer.forward(x)
        return x


class Autoencoder(pl.LightningModule):

    def __init__(self, hparams, encoder, decoder, train_set, val_set):
        super().__init__()
        # set hyperparams
        self.save_hyperparameters(hparams, ignore=['encoder', 'decoder'])

        # Define models
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.train_set = train_set
        self.val_set = val_set

    def forward(self, x):
        latent_repr, reconstruct_indices = self.encoder.forward(x)
        reconstruction = self.decoder.forward(latent_repr, reconstruct_indices)
        return reconstruction

    def general_step(self, batch, batch_idx, mode):
        images = batch
        flattened_images = images.view(images.shape[0], -1)

        # forward pass
        reconstruction = self.forward(flattened_images)

        # loss
        loss = F.mse_loss(reconstruction.reshape(-1, 28*28), flattened_images.reshape(-1, 28*28))

        return loss, reconstruction

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch, batch_idx, "train")
        self.log("train_loss_ae", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        flattened_images = images.view(images.shape[0], -1)

        reconstruction = self.forward(flattened_images)
        loss = F.mse_loss(reconstruction.reshape(-1, 28*28), flattened_images.reshape(-1, 28*28))

        reconstruction = reconstruction.view(
            reconstruction.shape[0], 28, 28).cpu().numpy()
        images = np.zeros((len(reconstruction), 3, 28, 28))
        for i in range(len(reconstruction)):
            images[i, 0] = reconstruction[i]
            images[i, 2] = reconstruction[i]
            images[i, 1] = reconstruction[i]
        self.logger.experiment.add_images(
            'reconstructions', images, self.current_epoch, dataformats='NCHW')
        return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'], num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'], num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()}
        ], lr=0.0001)
        return optimizer

    def getReconstructions(self, loader=None):
        self.eval()
        self = self.to(self.device)

        if not loader:
            loader = self.val_dataloader()

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(pl.LightningModule):

    def __init__(self, hparams, encoder, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.save_hyperparameters(hparams, ignore=['encoder'])
        self.encoder: nn.Sequential = encoder
        self.latent_dim = 30
        self.model = nn.Identity()
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, self.hparams["n_hidden_classifier"]),
            nn.LeakyReLU(),
            nn.Dropout(p=hparams["dropout_p_classifier"]),
            nn.Linear(self.hparams["n_hidden_classifier"], self.hparams["n_hidden_classifier"]),
            nn.LeakyReLU(),
            nn.Dropout(p=hparams["dropout_p_classifier"]),
            nn.Linear(self.hparams["n_hidden_classifier"], 10),
            nn.Softmax()
        )

    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch
        flattened_images = images.view(images.shape[0], -1)

        # forward pass
        out = self.forward(flattened_images)

        # loss
        loss = F.cross_entropy(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        self.log("train_loss_cls", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        self.log("val_loss", avg_loss)
        self.log("val_acc", acc)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()}
        ], lr=0.0001)
        return optimizer

    def getAcc(self, loader=None):
        self.eval()
        self = self.to(self.device)

        if not loader:
            loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
