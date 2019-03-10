import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class EncoderGRU(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0, bidirectional=False):
        super(EncoderGRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.bidirectional = bidirectional

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=bidirectional)

    def forward(self, input_seq, input_lengths, hidden=None):
        # input_seq: batch of input sentence, shape = (max_len, batch_size)
        # input_lengths = list of sentence lengths corresponding to each sent in the batch
        # hidden state of shape = (n_layers x num_directions, batch_size, hidden_size)

        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        # outputs: output features from the last hidden layer of GRU (sum of bidirectional outputs)
        # shape= (max_len, batch_size, hidden_size)
        return outputs, hidden