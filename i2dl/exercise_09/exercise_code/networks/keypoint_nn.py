"""Models for facial keypoint detection"""

import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class Residual(nn.Module):
    
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, X):
        Y = self.prelu1(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.prelu2(Y)


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""

    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()

        pretrained = torchvision.models.mobilenet_v3_large(pretrained=True)
        pretrained.classifier[3] = torch.nn.Linear(1280, 500)
                
        self.model = nn.Sequential(
            nn.Conv2d(1, 3, 1, 1),
            pretrained,
            nn.PReLU(),
            nn.Linear(500, 30)
        )

    def forward(self, x):
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        output = self.model(x)
        return output.reshape(-1, 30)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['keypoints'].reshape(-1, 30)
        output = self.forward(x)
        loss = F.mse_loss(output, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
        ], lr=3e-5, weight_decay=1e-4)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['keypoints'].reshape(-1, 30)
        output = self.forward(x)
        loss = F.mse_loss(output, y)
        self.log('val_loss', loss)
        return loss

class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
