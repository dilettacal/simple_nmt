import argparse
import torch

from torch import nn, optim

from tutorial.evaluate import GreedySearchDecoder, evaluateInput
from tutorial.train import run_experiment
from global_settings import FILENAME, DATA_DIR, SAVE_DIR, device
from data.prepro import *
from data.utils import train_split, filter_pairs
from data.tokenize import Vocab
from tutorial.decoder import DecoderGRU
from tutorial.encoder import EncoderGRU


if __name__ == '__main__':
    src_lang = "eng"
    trg_lang = "deu"
    exp_contraction = True
    src_reversed = False
    limit = 10000

    pairs = read_lines(os.path.join(".",DATA_DIR), FILENAME)
    print(len(pairs))
    print(pairs[10])

    cleaned_file = "%s-%s_cleaned" % (src_lang, trg_lang) + "_rude" if not exp_contraction else "%s-%s_cleaned" % (src_lang, trg_lang) + "_full"
    cleaned_file = cleaned_file + "reversed.pkl" if src_reversed else cleaned_file + ".pkl"

    #preprocess_pipeline(file, exp_contraction=exp_contraction)

   #if os.path.isfile(os.path.join(PREPRO_FILE, file)):
       # print("Loading existing file...")
        #pairs = load_cleaned_data(PREPRO_FILE, file)
    #else:
        #pairs = preprocess_pipeline(file, exp_contraction)

   # pairs = read_lines(os.path.join(DATA_DIR, FILENAME))
    print(os.path.join(DATA_DIR, cleaned_file))

    pairs = preprocess_pipeline(pairs, cleaned_file, exp_contraction)

    #pairs = filter_pairs(pairs, len_tuple=(1,15))

    if limit:
        pairs = pairs[:limit]

    print("Splitting train and test set...")
    train_set, test_set = train_split(pairs)
    save_clean_data(PREPRO_DIR, train_set, filename="train.pkl")
    save_clean_data(PREPRO_DIR, test_set, filename="test.pkl")
    print(train_set[:10])
    print(test_set[:10])
    print("Splitting src and trg sents for vocabularies...")
    src_sents = [item[0] for item in pairs]
    trg_sents = [item[1] for item in pairs]

    print(type(src_sents[1]))

    src_vocab = Vocab.build_vocab_from_pairs(src_sents, lang=src_lang)
    trg_vocab = Vocab.build_vocab_from_pairs(trg_sents, lang=trg_lang)

    print(src_vocab)
    print(trg_vocab)

    # Configure models
    model_name = 'nmt_model'
    hidden_size = 256
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoints to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 4000
    if os.path.isfile(os.path.join(SAVE_DIR, model_name, FILENAME,
                                '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                               '{}_checkpoint.tar'.format(checkpoint_iter))):

        loadFilename = os.path.join(SAVE_DIR, model_name, FILENAME,
                                '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                               '{}_checkpoint.tar'.format(checkpoint_iter))
    else:
        loadFilename = None

    loadFilename = None #comment this line if you want to use an existing model


    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoints = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        src_emb_check = checkpoint['src_embedding']
        trg_emb_check = checkpoint['trg_embedding']
        src_vocab.__dict__ = checkpoint['src_dict']
        trg_vocab.__dict__ = checkpoint['tar_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    src_emb = nn.Embedding(src_vocab.num_words, hidden_size)
    trg_emb = nn.Embedding(trg_vocab.num_words, hidden_size)

    if loadFilename:
        src_emb.load_state_dict(src_emb_check)
        trg_emb.load_state_dict(trg_emb_check)
    # Initialize encoder & decoder models
    encoder = EncoderGRU(hidden_size, src_emb, encoder_n_layers, dropout)
    decoder = DecoderGRU(trg_emb, hidden_size, trg_vocab.num_words, decoder_n_layers, dropout)

    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    print(encoder)
    print(decoder)


    # Configure training/optimization
    clip = 10.0
    teacher_forcing_ratio = 0.2
    learning_rate = 0.01
    decoder_learning_ratio = 5.0
    n_iteration = 30000
    print_every = 1000
    save_every = 1000

    # Ensure dropout layers are in train mode
    #encoder.train()
    #decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Running experiment.....")
    """
    def run_experiment(model_name, src_voc, tar_voc, encoder, decoder,
                   encoder_optimizer, decoder_optimizer,
                   src_embedding, trg_embedding,
                   encoder_n_layers, decoder_n_layers,
                   save_dir, n_iteration, batch_size, print_every,
                   save_every, clip, corpus_name, loadFilename, hidden_size, train_set_pairs, val_set_pairs, teacher_forcing_ratio=0):
    
    """
    run_experiment(model_name, src_vocab, trg_vocab, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   src_emb, trg_emb, encoder_n_layers, decoder_n_layers, SAVE_DIR, n_iteration, batch_size,
                   print_every, save_every, clip, "eng-deu.txt", loadFilename, hidden_size=hidden_size, teacher_forcing_ratio=teacher_forcing_ratio, train_set_pairs=train_set, val_set_pairs=test_set)




    print("Experiment complete!")
    print("\n")
    print("*"*100)
    print("Entering the translation mode....")

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)


    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, src_vocab, trg_vocab)