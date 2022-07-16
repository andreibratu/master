"""SegmentationNN"""
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.pretrained_model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            pretrained=True
        )
        # Replace final layer with a convolution layer with enough filters
        self.pretrained_model.classifier.low_classifier = torch.nn.Conv2d(40, 23, kernel_size=(1,1), stride=(1,1))
        self.pretrained_model.classifier.high_classifier = torch.nn.Conv2d(128, 23, kernel_size=(1,1), stride=(1,1))
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x = self.pretrained_model(x)['out']
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        losses = self.loss_func(outputs, targets)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.pretrained_model.parameters()},
        ], lr=3e-4, weight_decay=1e-4)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        losses = self.loss_func(outputs, targets)
        self.log('val_loss', losses)
        return losses
        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
