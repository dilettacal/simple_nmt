import os
import random

import torch
from torch import optim

from experiment.train_eval import evaluateInput, GreedySearchDecoder, trainIters
from global_settings import device, FILENAME, SAVE_DIR
from model.model import EncoderLSTM, DecoderLSTM
from utils.prepro import read_lines, preprocess_pipeline
from utils.tokenize import build_vocab

from global_settings import DATA_DIR


def define_model():
    pass

def run_experiment():
    pass

if __name__ == '__main__':


    src_lang = "eng"
    trg_lang = "deu"
    exp_contraction = True
    src_reversed = False
    limit = None

    start_root = "."
    pairs = read_lines(os.path.join(start_root, DATA_DIR), FILENAME)
    print(len(pairs))

    cleaned_file = "%s-%s_cleaned" % (src_lang, trg_lang) + "_rude" if not exp_contraction else "%s-%s_cleaned" % (
        src_lang, trg_lang) + "_full"
    cleaned_file = cleaned_file + "reversed.pkl" if src_reversed else cleaned_file + ".pkl"

    pairs = preprocess_pipeline(pairs, cleaned_file, exp_contraction)
    print(random.choice(pairs))

    limit = 10000

    if limit:
        pairs = pairs[:limit]

    # Build vocabularies
    src_sents = [item[0] for item in pairs]
    trg_sents = [item[1] for item in pairs]

    input_lang = build_vocab(src_sents, "eng")
    output_lang = build_vocab(trg_sents, "deu")


    # Configure models
    model_name = 'simple_nmt_model'
    hidden_size = 256
    encoder_n_layers = 1
    decoder_n_layers = 1
    dropout = 0.1
    batch_size = 64
    input_size = input_lang.num_words
    output_size = output_lang.num_words
    embedding_size = 256

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 4000
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))

    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        src_embed = checkpoint['src_emb']
        trg_embed = checkpoint['trg_emb']
        input_lang.__dict__ = checkpoint['src_voc']
        output_lang.__dict__ = checkpoint['trg_voc']

    print('Building encoder and decoder ...')
    # Initialize encoder & decoder models
    encoder = EncoderLSTM(input_size=input_size, emb_size=embedding_size, hidden_size=hidden_size,
                         n_layers=encoder_n_layers, dropout=dropout)
    decoder = DecoderLSTM(output_size=output_size, emb_size=embedding_size, hidden_size=hidden_size, n_layers= decoder_n_layers)

    if loadFilename:
        src_emb = encoder.embedding
        trg_emb = decoder.embedding
        src_emb.load_state_dict(src_embed)
        trg_emb.load_state_dict(trg_embed)


    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built:')
    print(encoder)
    print(decoder)


    # Configure training/optimization
    clip = 30.0
    teacher_forcing_ratio = 0.3
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 10000
    print_every = 100
    save_every = 500

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    trainIters(model_name, input_lang, output_lang, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               encoder_n_layers, decoder_n_layers, SAVE_DIR, n_iteration, batch_size,
               print_every, save_every, clip, FILENAME, loadFilename)


    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    evaluateInput(encoder, decoder, searcher, input_lang, output_lang)