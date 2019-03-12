import argparse

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from data.prepro import *
from data.tokenize import Vocab, NMTDataset, Tokenizer, collate_fn
from data.utils import train_split
from global_settings import FILENAME, DATA_DIR, device
from model.nmtmodel import NMTModel, count_parameters, EncoderLSTM, DecoderLSTM
from experiment.experiment import run_experiment

if __name__ == '__main__':

    src_lang = "eng"
    trg_lang = "deu"
    exp_contraction = True
    src_reversed = False
    limit = None

    start_root = "."
    pairs = read_lines(os.path.join(start_root, DATA_DIR), FILENAME)
    print(len(pairs))
    print(pairs[10])

    cleaned_file = "%s-%s_cleaned" % (src_lang, trg_lang) + "_rude" if not exp_contraction else "%s-%s_cleaned" % (
    src_lang, trg_lang) + "_full"
    cleaned_file = cleaned_file + "reversed.pkl" if src_reversed else cleaned_file + ".pkl"

    # preprocess_pipeline(file, exp_contraction=exp_contraction)

    # if os.path.isfile(os.path.join(PREPRO_FILE, file)):
    # print("Loading existing file...")
    # pairs = load_cleaned_data(PREPRO_FILE, file)
    # else:
    # pairs = preprocess_pipeline(file, exp_contraction)

    # pairs = read_lines(os.path.join(DATA_DIR, FILENAME))
    #print(os.path.join(DATA_DIR, cleaned_file))

    pairs = preprocess_pipeline(pairs, cleaned_file, exp_contraction)

    if limit:
        pairs = pairs[:limit]


    print("Building tokenizer objects....")
    src_tokenizer = Tokenizer([item[0] for item in pairs], src_lang)
    trg_tokenizer = Tokenizer([item[1] for item in pairs], src_lang)

    print("Vocabularies overview:")
    print("Source words: ", src_tokenizer.vocab.num_words)
    print("Target words:", trg_tokenizer.vocab.num_words)

    print("Building the dataset...")

    dataset = NMTDataset(pairs, "eng", "deu", src_tokenizer, trg_tokenizer)
    print("Dataset example: ")
    print(dataset.__getitem__(50))

    print("Preparing dataset splits...")
    train_set, test_set = train_split(pairs)
    dataset.set_split('train', train_set)
    dataset.set_split('test', test_set)

    print("Total train samples:", dataset.train.__len__())
    print("Total validation samples:", dataset.test.__len__())

    print("Persisting splittings...")
    save_clean_data(PREPRO_DIR, train_set, filename="train.pkl")
    save_clean_data(PREPRO_DIR, test_set, filename="test.pkl")

    BATCH_SIZE = 24

    print("Building dataloaders. Batch size: %s", BATCH_SIZE)

    train_iter = DataLoader(dataset=dataset.train,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            #num_workers=4,
                            collate_fn=collate_fn)


    test_iter = DataLoader(dataset=dataset.test, batch_size=1, shuffle=False, collate_fn=collate_fn)
    #test_iter: array with (src_sent), (trg_sent), [tensor([14]), tensor([35]), tensor([2126]), tensor([76]), tensor([177]), tensor([2126]), tensor([2])]...

    #src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch


    print("Set up the model...")
    # Configure models
    model_name = 'nmt_model'

    INPUT_DIM = src_tokenizer.vocab.num_words
    OUTPUT_DIM = trg_tokenizer.vocab.num_words
    EMBEDDING_DIMENSION = 256
    HIDDEN_SIZE = 512
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0
    EPOCHS = 1

    enc = EncoderLSTM(vocab_size=INPUT_DIM, emb_dim=EMBEDDING_DIMENSION, rnn_hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=ENC_DROPOUT)
    dec = DecoderLSTM(vocab_size=OUTPUT_DIM, emb_dim=EMBEDDING_DIMENSION, rnn_hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS, dropout=DEC_DROPOUT)
    model = NMTModel(enc, dec, device).to(device)

    print("Model architecture overview: ")
    print(model)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)


    print("Starting experiment...")
    run_experiment(src_voc=src_tokenizer.vocab, tar_voc=trg_tokenizer.vocab, model=model, optimizer=optimizer, num_epochs=EPOCHS, criterion=criterion,
                   train_set=train_set, eval_set=test_set, train_batch_size=BATCH_SIZE, val_batch_size=BATCH_SIZE, clip=10., teacher_forcing_ratio=0.2, train_iteration = train_n_iterations, val_iteration=val_n_iterations)







