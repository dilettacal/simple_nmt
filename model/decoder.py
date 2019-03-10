import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class DecoderGRU(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        output, hidden = self.gru(embedded, last_hidden)
        output, hidden = self.gru(output, hidden)  #

        # Squeeze first dimension
        output = output.squeeze(0)
        # Prediction
        output = self.out(output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
