import os
import random
from pickle import load

import torch
from torch import optim

from experiment.train_eval import evaluateInput, GreedySearchDecoder, trainIters
from global_settings import device, FILENAME, SAVE_DIR, PREPRO_DIR, TRAIN_FILE, TEST_FILE, EXPERIMENT_DIR, LOG_FILE
from model.model import EncoderLSTM, DecoderLSTM
from utils.prepro import read_lines, preprocess_pipeline, load_cleaned_data
from utils.tokenize import build_vocab, tensor_max_len

from global_settings import DATA_DIR
from utils.utils import split_data, filter_pairs, max_length


def define_model():
    pass

def run_experiment():
    pass

if __name__ == '__main__':

    start_root = "."
    src_lang = "eng"
    trg_lang = "deu"
    exp_contraction = True
    src_reversed = False
    limit = None

    cleaned_file = "%s-%s_cleaned" % (src_lang, trg_lang) + "_rude" if not exp_contraction else "%s-%s_cleaned" % (
        src_lang, trg_lang) + "_full"
    cleaned_file = cleaned_file + "reversed.pkl" if src_reversed else cleaned_file + ".pkl"

    if os.path.isfile(os.path.join(PREPRO_DIR, cleaned_file)):
        print("File already preprocessed! Loading file....")
        pairs = load_cleaned_data(PREPRO_DIR, filename=cleaned_file)
    else:
        print("No preprocessed file found. Starting data preprocessing...")
        pairs = read_lines(os.path.join(start_root, DATA_DIR), FILENAME)
        pairs, path = preprocess_pipeline(pairs, cleaned_file, exp_contraction) #data/prepro/eng-deu_cleaned_full.pkl


    print("Sample from data:")
    print(random.choice(pairs))

    limit = 30000

    if limit:
        pairs = pairs[:limit]

    train_set, test_set = split_data(pairs)

    print("Data in train set:", len(train_set))
    print("Data in test set:", len(test_set))

    print("Building vocabularies...")
    # Build vocabularies based on dataset
    src_sents = [item[0] for item in pairs]
    trg_sents = [item[1] for item in pairs]

    max_src_l = max_length(src_sents)
    max_trg_l = max_length(trg_sents)

    print("Max sentence length in source sentences:", max_src_l)
    print("Max sentence length in source sentences:", max_trg_l)

    input_lang = build_vocab(src_sents, "eng")
    output_lang = build_vocab(trg_sents, "deu")

    # Configure models
    model_name = ''
    model_name += 'simple_nmt_model'+str(limit) if limit else 'simple_nmt_model'
    hidden_size = 256
    encoder_n_layers = 1
    decoder_n_layers = 1
    batch_size = 64
    input_size = input_lang.num_words
    output_size = output_lang.num_words
    embedding_size = 256

    print('Building encoder and decoder ...')
    encoder = EncoderLSTM(input_size=input_size, emb_size=embedding_size, hidden_size=hidden_size,
                         n_layers=encoder_n_layers)
    decoder = DecoderLSTM(output_size=output_size, emb_size=embedding_size, hidden_size=hidden_size, n_layers= decoder_n_layers)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built:')
    print(encoder)
    print(decoder)

    clip = 30.0
    teacher_forcing_ratio = 0.3
    learning_rate = 0.001 #0.0001
    decoder_learning_ratio = 2.0 #5.0
    n_iteration = 10000
    print_every = 100
    save_every = 500

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # Run training iterations
    print("Starting Training!")
    val_loss, directory = trainIters(model_name, input_lang, output_lang, train_set, test_set, encoder, decoder, encoder_optimizer, decoder_optimizer,
               encoder_n_layers, decoder_n_layers, SAVE_DIR, n_iteration, batch_size,
               print_every, save_every, clip, FILENAME)

    print("Checkponits saved in %s" %(directory))

    #Log file name
    try:
        with open(os.path.join(start_root, EXPERIMENT_DIR, LOG_FILE), encoding="utf-8", mode="w") as f:
            print("Logging to file...")
            f.write("Experiment name:\n")
            f.write(model_name)
            f.write("\nDirectory:\n")
            f.write(str(directory))
            f.write("\n Max src length %s" %max_src_l)
            f.write("\n Max trg length %s"%max_trg_l)
            f.write("\nAverage validation loss:\n")
            f.write(str(val_loss))
            f.write("\n**********************************")
    except IOError or TypeError or RuntimeError:
        print("Log to file failed!")

