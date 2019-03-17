import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Model components:
Encoder and Decoder based on LSTMs.
Code inpsired by PyTorch Chatbot Tutorial:https://pytorch.org/tutorials/beginner/chatbot_tutorial.html


"""

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, n_layers=1, dropout=0.5, bidirectional=False):
        super(EncoderLSTM, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional


        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers,
                           dropout=(dropout if n_layers > 1 else 0), bidirectional=bidirectional)
        self.dropout= nn.Dropout(dropout)

    def forward(self, input_seq, input_lengths):
        # input_seq: batch of input sentence, shape = (max_len, batch_size)
        # input_lengths = list of sentence lengths corresponding to each sent in the batch
        # hidden state of shape = (n_layers x num_directions, batch_size, hidden_size)

        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.lstm(packed)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        # outputs: output features from the last hidden layer of GRU (sum of bidirectional outputs)
        # shape= (max_len, batch_size, hidden_size)
        return outputs, hidden


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, n_layers=1, dropout=0.2):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Define layers
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, dropout=(dropout if n_layers > 1 else 0))
        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_step, last_hidden):
        #input_step = [seq_len, batch_size]
        #last_hidden = [seq_len, batch_size, hidden_size] #1, 64, 256
        #embedded = [seq_len, batch_size, embedding_size]
        embedded = self.embedding(input_step)
        embedded = self.dropout(embedded)

        # Forward through unidirectional GRU
        output, hidden = self.lstm(embedded, last_hidden)
        # Squeeze first dimension
        output = output.squeeze(0)
        # Prediction
        output = self.out(output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
