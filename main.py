import argparse
import torch
import os

from torch import nn, optim

from experiment.evaluate import GreedySearchDecoder, evaluateInput
from experiment.train import trainIters
from global_settings import FILENAME, DATA_DIR, EXPERIMENT_DIR, PREPRO_DIR, SAVE_DIR, device
from data.prepro import *
from data.utils import train_split
from data.tokenize import Vocab
from model.decoder import DecoderGRU
from model.encoder import EncoderGRU

parser = {
    'epochs': 50,
    'batch_size': 100,
    'max_len': 20,  # max length of grapheme/phoneme sequences
    'src_emb_dim': 500,  # embedding dimension
    'trg_emb_dim': 500,
    'hidden_dim': 500,  # hidden dimension
    'log_every': 100,  # number of iterations to log and validate training
    'lr': 0.007,  # initial learning rate
    'clip': 2.3,  # clip gradient, to avoid exploding gradient
    'cuda': True,  # using gpu or not
    'seed': 5,  # initial seed
    'store_dir': './experiment/checkpoint/',  # path to save models
    'data_dir': './data/',
    'prepro_file':'./data/prepro/',
    'limit': None, # get the first 'limit' pairs of the dataset -> reduces dataset
    'min_sent_len': None,
    'max_sent_len':None,
    'filter_question': False,
    'filter_eng_prefixes': False,
    'expand_contractions': True
}

def preprocess_pipeline(pairs, cleaned_file_to_store, exp_contraction=None, reverse_pairs=False):
    """
    Assuming english as input language
    :param cleaned_file_to_store:
    :param exp_contraction:
    :param reverse_pairs:
    :return:
    """

   # pairs = read_lines(DATA_DIR, FILENAME)
  #  print(len(pairs))
    #print(pairs[10])
    src_sents = [item[0] for item in pairs]
    trg_sents = [item[1] for item in pairs]
    if expand_contraction:
        src_mapping = ENG_CONTRACTIONS_MAP
        trg_mapping = UMLAUT_MAP
    else:
        src_mapping = exp_contraction
        trg_mapping = exp_contraction

    cleaned_pairs = []
    for i, (src_sent, trg_sent) in enumerate(pairs):
        cleaned_src_sent = preprocess_sentence(src_sent, src_mapping)
        cleaned_trg_sent = preprocess_sentence(trg_sent, trg_mapping)
        cleaned_list = [cleaned_src_sent, cleaned_trg_sent]

        cleaned_pairs.append(cleaned_list)

    if reverse_pairs:
        cleaned_pairs = reverse_language_pair(cleaned_pairs)

    save_clean_data(PREPRO_DIR, cleaned_pairs, cleaned_file_to_store)
    return cleaned_pairs



if __name__ == '__main__':
    args = argparse.Namespace(**parser)
    args.cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    src_lang = "eng"
    trg_lang = "deu"
    exp_contraction = False
    src_reversed = False
    limit = 10000

    pairs = read_lines(DATA_DIR, FILENAME)
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
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
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
    #TODO: Re-train model on train_set
    trainIters(model_name, src_vocab, trg_vocab, train_set, encoder, decoder, encoder_optimizer, decoder_optimizer,
               src_emb, trg_emb, encoder_n_layers, decoder_n_layers, SAVE_DIR, n_iteration, batch_size,
               print_every, save_every, clip, "eng-deu.txt", loadFilename, hidden_size=hidden_size, teacher_forcing_ratio=teacher_forcing_ratio)


    #TODO: Implement evaluation on test or random set and then start evaluation at runtime

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)


    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, src_vocab, trg_vocab)