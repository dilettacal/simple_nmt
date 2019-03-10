import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 rnn_hidden_size, n_layers=1, dropout = 0,
                 bidirectional=False):
        super(EncoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        #Layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim,padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, rnn_hidden_size, num_layers=n_layers, bidirectional=bidirectional)

    def forward(self, input_seq, input_lengths):
        seq_embedded = self.dropout(self.embedding(input_seq))
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(seq_embedded, input_lengths)

        enc_outputs, (enc_hidden, enc_cell) = self.rnn(seq_packed)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_outputs)

        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, enc_hidden, enc_cell