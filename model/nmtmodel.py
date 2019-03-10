import random

import torch
import torch.nn as nn
from global_settings import device
from model.decoder import DecoderLSTM
from model.encoder import EncoderLSTM


class NMTModel(nn.Module):
    def __init__(self, encoder:EncoderLSTM, decoder:DecoderLSTM, device=device):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def forward(self, src_input, trg_input, teacher_forcing_ratio=0.3, src_lengths=None):

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



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)