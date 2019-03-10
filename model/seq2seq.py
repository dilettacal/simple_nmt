import random

import torch
import torch.nn as nn
from global_settings import device
from model.decoder import DecoderLSTM
from model.encoder import EncoderLSTM


class Seq2Seq(nn.Module):
    def __init__(self, encoder:EncoderLSTM, decoder:DecoderLSTM, searcher, device=device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.searcher = searcher

    def forward(self, src_input, trg_input, teacher_forcing_ratio=0.3):
        batch_size = trg_input.shape[1]
        seq_len = trg_input.shape[0]

        output_size = self.decoder.vocab_size

        dec_outputs = torch.zeros(seq_len, batch_size, output_size).to(device)

        #Encode the input sequences

        output, hidden, cell = self.encoder(src_input)

        decoder_input = self.decoder.init_input(batch_size)
        hidden = hidden[:self.decoder.n_layers]

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False


        #Run through target row by row
        for timestep in range(seq_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            #store results
            dec_outputs[timestep] = output
            if use_teacher_forcing:
                # determine teacher forced input
                decoder_input = trg_input[timestep].view(1,-1) #real target value
            else:
                #Use prediction
                _, topi = output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
        return dec_outputs

