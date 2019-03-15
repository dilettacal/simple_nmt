import random

import torch
import torch.nn as nn
from torch import nn as nn

from global_settings import device


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, rnn_hidden_size, n_layers=1, dropout = 0, bidirectional=False):
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

    def forward(self, input_seq, input_lengths=None):

        seq_embedded = self.dropout(self.embedding(input_seq))

        if not input_lengths is None:
            seq_embedded = torch.nn.utils.rnn.pack_padded_sequence(seq_embedded, input_lengths)

            enc_outputs, (enc_hidden, enc_cell) = self.rnn(seq_embedded)

            enc_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_outputs)
        else:
            enc_outputs, (enc_hidden, enc_cell) = self.rnn(seq_embedded)

        return enc_outputs, enc_hidden, enc_cell


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, rnn_hidden_size, sos_idx=1, n_layers=1, dropout=0):
        super(DecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.sos_idx = sos_idx
        self.n_layers = n_layers

        # Layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, rnn_hidden_size, num_layers=n_layers, dropout=dropout)
        self.out = nn.Linear(rnn_hidden_size, vocab_size)


    def init_input(self, batch_size):
        #Initializes first vector with only SOS idx (1)
        #torch.LongTensor([[1 for _ in range(small_batch_size)]])
        return torch.LongTensor([[self.sos_idx for _ in range(batch_size)]]).to(device)

        #input_step, last_hidden, encoder_outputs
    def forward(self, input_step, last_hidden, context):
        #input_step shape [1,batch_size]
        embedded = self.dropout(self.embedding(input_step))

        output, (hidden, cell) = self.rnn(embedded, (last_hidden, context))

        output= output.squeeze(0)

        output = self.out(output)

        return output, hidden, cell


class NMTModel(nn.Module):
    def __init__(self, encoder:EncoderLSTM, decoder:DecoderLSTM, device=device):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def forward(self, src_input, trg_input, teacher_forcing_ratio=0.3, src_lengths=None):

        #Sequences are stored as (seq_len, batch_size)
        #Batch size is the same for both input and target
        #seq_len can differ. As the sequence in the decoding phase is processed one timestep at time
        #seq_len is set to the max length in the target batch

        batch_size = trg_input.shape[1]
        seq_len = trg_input.shape[0]

        output_size = self.decoder.vocab_size

        dec_outputs = torch.zeros(seq_len, batch_size, output_size).to(device)

        src_input = src_input.to(device)

        #Encode the input sequences

        output, hidden, cell = self.encoder(src_input, src_lengths)

        decoder_input = self.decoder.init_input(batch_size) #SOS
        #hidden = hidden[:self.decoder.n_layers]

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False


        #Run through target row by row
        for timestep in range(seq_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            #store results
            dec_outputs[timestep] = output

            """
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
            """

            if use_teacher_forcing:
                # determine teacher forced input
                decoder_input = trg_input[timestep].view(1,-1) #real target value
            else:
                #Use prediction
                _, topi = output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
        return dec_outputs


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, (encoder_hidden, cell) = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

