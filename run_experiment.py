import argparse

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from data.prepro import *
from data.tokenize import Vocab, NMTDataset, Tokenizer, collate_fn
from data.utils import train_split
from global_settings import FILENAME, DATA_DIR, device
from model.decoder import DecoderLSTM
from model.encoder import EncoderLSTM
from model.nmtmodel import NMTModel, count_parameters
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

    src_tokenizer = Tokenizer([item[0] for item in pairs], src_lang)
    trg_tokenizer = Tokenizer([item[1] for item in pairs], src_lang)


    dataset = NMTDataset(pairs, "eng", "deu", src_tokenizer, trg_tokenizer)

    print(dataset.__getitem__(1))

    train_set, test_set = train_split(pairs)
    dataset.set_split('train', train_set)
    dataset.set_split('test', test_set)

    save_clean_data(PREPRO_DIR, train_set, filename="train.pkl")
    save_clean_data(PREPRO_DIR, test_set, filename="test.pkl")

    batch_size = 48

    train_iter = DataLoader(dataset=dataset.train,
                            batch_size=batch_size,
                            shuffle=True,
                            #num_workers=4,
                            collate_fn=collate_fn)

    for i, batch in enumerate(train_iter):
        if i == 0:
            src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch
            print(src_seqs.shape)
            print(tgt_seqs.shape)
    exit()
    #print(train_set[:10])
    #print(test_set[:10])
    print("Creating vocabularies...")
    src_sents = [item[0] for item in pairs]
    trg_sents = [item[1] for item in pairs]
    src_vocab = Vocab.build_vocab_from_pairs(src_sents, lang=src_lang)
    trg_vocab = Vocab.build_vocab_from_pairs(trg_sents, lang=trg_lang)
    print(src_vocab)
    print(trg_vocab)

    # Configure models
    model_name = 'nmt_model'

    INPUT_DIM = src_vocab.num_words
    OUTPUT_DIM = trg_vocab.num_words
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    BATCH_SIZE = 128
    train_n_iterations = int(len(train_set) / BATCH_SIZE)
    val_n_iterations = int(len(test_set) / BATCH_SIZE)

    EPOCH = 5

    TOTAL_ITERATIONS = EPOCH * train_n_iterations

    enc = EncoderLSTM(vocab_size=INPUT_DIM, emb_dim=ENC_EMB_DIM, rnn_hidden_size=HID_DIM, n_layers=N_LAYERS, dropout=ENC_DROPOUT)
    dec = DecoderLSTM(vocab_size=OUTPUT_DIM, emb_dim=DEC_EMB_DIM, rnn_hidden_size=HID_DIM, n_layers=N_LAYERS, dropout=DEC_DROPOUT)
    model = NMTModel(enc, dec, device).to(device)

    print("Model architecture summary: ")
    print(model)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)


    print("Starting experiment...")
    run_experiment(src_voc=src_vocab, tar_voc=trg_vocab, model=model, optimizer=optimizer, num_epochs=EPOCH, criterion=criterion,
                   train_set=train_set, eval_set=test_set, train_batch_size=BATCH_SIZE, val_batch_size=BATCH_SIZE, clip=10., teacher_forcing_ratio=0.2, train_iteration = train_n_iterations, val_iteration=val_n_iterations)







