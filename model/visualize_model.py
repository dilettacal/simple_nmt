import random

from torch import optim, nn

from model.nmtmodel import NMTModel, EncoderLSTM, DecoderLSTM

from data.tokenize import batch2TrainData
from global_settings import FILENAME, DATA_DIR, SAVE_DIR, device
from data.prepro import *
from data.utils import train_split
from data.tokenize import Vocab
import torch

if __name__ == '__main__':
    exp_contraction = True
    src_reversed = False
    limit = None
    src_lang = "eng"
    trg_lang = "deu"

    start_root = "../"

    pairs = read_lines(os.path.join(start_root, DATA_DIR), FILENAME)
    print(pairs[10])
    pairs = preprocess_pipeline(pairs, exp_contraction=exp_contraction)
    print(pairs[10])

    pairs = pairs[:1000]
    src_sents = [item[0] for item in pairs]
    trg_sents = [item[1] for item in pairs]
    src_vocab = Vocab.build_vocab_from_pairs(src_sents, lang=src_lang)
    trg_vocab = Vocab.build_vocab_from_pairs(trg_sents, lang=trg_lang)

    small_batch_size = 5
    batches = batch2TrainData(src_vocab, trg_vocab, [random.choice(pairs) for _ in range(small_batch_size)])
    input_var, lengths, tar_var, tar_length, max_tar_len = batches
    print("Input var shape:", input_var.shape)
    print("Lengths:", lengths.shape)
    print("Tar var shape:", tar_var.shape)
    print("Tar len shape:", tar_length.shape)

    hidden_size = 256
    n_layers = 2
    dropout = 0.1

    encoder = EncoderLSTM(vocab_size=src_vocab.num_words, emb_dim=hidden_size, rnn_hidden_size=hidden_size, n_layers=n_layers, dropout=dropout)
    decoder = DecoderLSTM(vocab_size=trg_vocab.num_words, emb_dim=hidden_size, rnn_hidden_size=hidden_size, n_layers=n_layers)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print(encoder)
    print(decoder)

    # Dropout layers in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    encoder_optim = optim.Adam(encoder.parameters(), lr=0.0001)
    decoder_optim = optim.Adam(decoder.parameters(), lr=0.0001)

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    input_var = input_var.to(device)
    lengths = lengths.to(device)
    tar_var = tar_var.to(device)
    tar_length = tar_length.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outs, enc_hidden, enc_cell = encoder(input_var, lengths)
    print("Encoder output shapes:", encoder_outs.shape)
    print("Last encoder hidden state:", enc_hidden.shape)
    print("Last encoder cell state:", enc_cell.shape)

    decoder_input = decoder.init_input(tar_var.shape[1])
    print(tar_var.shape[1])
    decoder_input = decoder_input.to(device)
    print("INitial decoder input shape:", decoder_input.shape)
    print(decoder_input)
    standard_input = torch.LongTensor([[1 for _ in range(small_batch_size)]])
    print(standard_input)

    decoder_hidden = enc_hidden[:decoder.n_layers]
    decoder_context = decoder_hidden
    print(decoder_hidden.shape)

    print("Visualize timesteps in the RNN:\n")

    print(max_tar_len == tar_var.shape[0])

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(max_tar_len):
        print("*" * 20, "Start computation for time step %s" % t, "*" * 20)
        #encoder_state, last_enc_hidden, last_enc_cell, target_seq
        decoder_output, decoder_hidden, decoder_context = decoder(decoder_input, decoder_hidden, decoder_context)
        print("Decoder output shape:", decoder_output.shape)
        print("Decoder hidden shape:", decoder_hidden.shape)
        # Teacher forcing: Next input is the current label
        decoder_input = tar_var[t].view(1, -1)
        print("Target variable before reshaping (shape):", tar_var[t].shape)
        print("Target variable before reshaping:", tar_var[t])
        print("Target variable after reshaping (new decoder input):", decoder_input.shape)

        loss = criterion(decoder_output, tar_var[t])

        encoder_optim.step()
        decoder_optim.step()

        print("Loss: ", loss.item())

        print("*" * 20, "Done with this timestep!", "*" * 20)
        print("\n")

        """
        Output example:
        ******************** Start computation for time step 1 ********************
        Decoder output shape: torch.Size([5, 698])
        Decoder hidden shape: torch.Size([2, 5, 256])
        Target variable before reshaping (shape): torch.Size([5])
        Target variable before reshaping: tensor([ 23, 165,  86,  86,   2], device='cuda:0')
        Target variable after reshaping (new decoder input): torch.Size([1, 5])
        Loss:  6.553864479064941
        ******************** Done with this timestep! ********************
        
        """


