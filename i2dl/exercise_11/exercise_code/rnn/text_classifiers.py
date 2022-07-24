import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import pytorch_lightning as pl

from exercise_code.rnn.rnn_nn import Embedding

class RNNClassifier(pl.LightningModule):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }
        self.num_layers = 4
        self.hidden_dim = hidden_size
        # if you do not inherit from lightning module use the following line
        # self.hparams = hparams
        
        # if you inherit from lightning module, comment out the previous line and use the following line
        self.hparams.update(hparams)

        self.loss_func = nn.BCELoss()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, 0)
        
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, 3, dropout=0.5)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.PReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################
        embedded = self.embedding(sequence)
        if lengths is not None:
            embedded = pack_padded_sequence(embedded, lengths)

        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        lstm_out, _ = self.lstm.forward(
            embedded
        )
        if lengths is not None:
            lstm_out = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)[0]
        
        # assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        sig_out = self.classifier.forward(
            lstm_out[-1, :, :]
        )
        # print('FINAL', sig_out.shape, sequence.shape)
        return sig_out.reshape(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.embedding.parameters()},
            {"params": self.lstm.parameters()},
            {"params": self.classifier.parameters()},
        ], lr=3e-4, weight_decay=1e-4)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # print('TRAIN')
        inputs, targets, lengths = batch['data'], batch['label'], batch['lengths']
        
        outputs = self.forward(inputs, lengths)

        # print('VTM', outputs.shape, targets.shape)
        losses = self.loss_func(outputs, targets)
        self.log('val_loss', losses)
        return losses

    def training_step(self, batch, batch_idx):
        # print('VALID')
        inputs, targets, lengths = batch['data'], batch['label'], batch['lengths']
       
        outputs = self.forward(inputs, lengths)
        
        losses = self.loss_func(outputs, targets)
        self.log('train_loss', losses)
        return losses
