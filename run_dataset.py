import os
import random
import time
from pickle import load

import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from experiment.train_eval import evaluateInput, GreedySearchDecoder, trainIters
from global_settings import device, FILENAME, SAVE_DIR, PREPRO_DIR, TRAIN_FILE, TEST_FILE
from model.model import EncoderLSTM, DecoderLSTM
from utils.prepro import read_lines, preprocess_pipeline, load_cleaned_data
from utils.tokenize import build_vocab, CustomDataset, pad_sequences, max_length, SOS

from global_settings import DATA_DIR
from utils.utils import split_data, filter_pairs, sort_batch, loss_function


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
        print(len(pairs))

        pairs, path = preprocess_pipeline(pairs, cleaned_file, exp_contraction) #data/prepro/eng-deu_cleaned_full.pkl


    print("Sample from data:")
    print(random.choice(pairs))

    limit = None

    if limit:
        pairs = pairs[:limit]


    # Build vocabularies
    src_sents = [item[0] for item in pairs]
    trg_sents = [item[1] for item in pairs]

    input_lang = build_vocab(src_sents, "eng")
    output_lang = build_vocab(trg_sents, "deu")

    #TODO: Google works on samples which are stored with SOS and EOS (both input and target)

    input_tensor = input_lang.tensorize_sequence_list(src_sents, append_sos=True, append_eos=True)
    target_tensor = output_lang.tensorize_sequence_list(trg_sents, append_sos=True, append_eos=True)
    print(pairs[10])
    print(input_tensor[10], target_tensor[10])

    print("Inplace padding...")
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    input_tensor = [pad_sequences(x, max_length_inp) for x in input_tensor]
    target_tensor = [pad_sequences(x, max_length_tar) for x in target_tensor]


    print("Splitting data...")

    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)

    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


    ### Setup
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_inp_size = input_lang.num_words
    vocab_tar_size = output_lang.num_words

    print("Creating dataloader...")

    train_dataset = CustomDataset(input_tensor_train, target_tensor_train)
    val_dataset = CustomDataset(input_tensor_val, target_tensor_val)

    dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                         drop_last=True,
                         shuffle=True)

    validation = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)

    first_batch = next(iter(dataset))
    print(first_batch)


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

    criterion = CrossEntropyLoss()

    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=0.001)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        start = time.time()

        encoder.train()
        decoder.train()

        total_loss = 0

        for (batch, (inp, targ, inp_len)) in enumerate(dataset):
            loss = 0

            xs, ys, lens = sort_batch(inp, targ, inp_len)
            enc_output, enc_hidden, enc_cell = encoder(xs.to(device), lens)
            dec_hidden = enc_hidden

            # use teacher forcing - feeding the target as the next input (via dec_input)
            dec_input = torch.tensor([[output_lang.word2index[SOS]]] * BATCH_SIZE)

            # run code below for every timestep in the ys batch
            for t in range(1, ys.size(1)):
                predictions, dec_hidden, _ = decoder(dec_input.to(device),
                                                     dec_hidden.to(device),
                                                     enc_output.to(device))
                loss += loss_function(ys[:, t].to(device), predictions.to(device))
                # loss += loss_
                dec_input = ys[:, t].unsqueeze(1)

            batch_loss = (loss / int(ys.size(1)))
            total_loss += batch_loss

            optimizer.zero_grad()

            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'
                      .format(epoch + 1, batch,batch_loss.detach().item()))

        encoder.eval()
        decoder.eval()

        total_valid_loss = 0

        ### Validate
        with torch.no_grad():
            for (batch, (inp, targ, inp_len)) in enumerate(validation):
                loss = 0

                xs, ys, lens = sort_batch(inp, targ, inp_len)
                enc_output, enc_hidden = encoder(xs.to(device), lens, device)
                dec_hidden = enc_hidden

                # use teacher forcing - feeding the target as the next input (via dec_input)
                dec_input = torch.tensor([[output_lang.word2idx['<SOS>']]] * BATCH_SIZE)

                # run code below for every timestep in the ys batch
                for t in range(1, ys.size(1)):
                    predictions, dec_hidden, _ = decoder(dec_input.to(device),
                                                         dec_hidden.to(device),
                                                         enc_output.to(device))
                    loss += loss_function(ys[:, t].to(device), predictions.to(device))
                    # loss += loss_
                    dec_input = ys[:, t].unsqueeze(1)

                batch_loss = (loss / int(ys.size(1)))
                total_valid_loss += batch_loss


            if batch % 100 == 0:
                print('Epoch {} Batch {} Valid Loss {:.4f}'
                      .format(epoch + 1, batch, batch_loss.detach().item()))

        ### TODO: Save checkpoint for model
        print('Epoch {} Train Loss {:.4f} Validation Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH, total_valid_loss/N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
