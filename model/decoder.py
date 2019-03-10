import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, rnn_hidden_size, sos_idx=1, n_layers=1, dropout=0.1):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.sos_idx = sos_idx
        self.n_layers = n_layers

        # Layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, rnn_hidden_size, num_layers=n_layers, dropout=(0 if n_layers==1 else dropout))
        self.out = nn.Linear(rnn_hidden_size, vocab_size)


    def init_input(self, batch_size):
        #Initializes first vector with only SOS idx (1)
        #torch.LongTensor([[1 for _ in range(small_batch_size)]])
        return torch.LongTensor([[self.sos_idx for _ in range(batch_size)]])

        #input_step, last_hidden, encoder_outputs
    def forward(self, input_step, last_hidden, context):
        embedded = self.dropout(self.embedding(input_step))

        output, (hidden, cell) = self.rnn(embedded, (last_hidden, context))

        output= output.squeeze(0)

        output = self.out(output)

        return output, hidden, cell
