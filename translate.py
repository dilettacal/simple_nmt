"""
Loads the model and translate given a keyboard input
experiment/checkpoints/simple_nmt_model30000/deu.txt/1-1_256

"""
import argparse
import os
import random

import torch
from torch import nn, optim

from experiment.train_eval import GreedySearchDecoder, evaluateInput
from global_settings import EXPERIMENT_DIR, LOG_FILE, device
from model.model import EncoderLSTM, DecoderLSTM
from utils.tokenize import Voc


def translate(start_root, path=None):
    # start_root = "."

    print("Reading experiment information from: ")
    if not path:
        try:
            with open(os.path.join(start_root, EXPERIMENT_DIR, LOG_FILE), mode="r", encoding="utf-8") as f:
                lines = f.readlines()
                print("Lines:", lines)

        except IOError or FileNotFoundError or FileExistsError or RuntimeError:
            print("Logging file does not exist!")
            exit(-1)
    else:
        lines= [path]

    experiment_path = [line.strip() for line in lines if "experiment/" in line]
    print(experiment_path[0])

    # e.g. ['1-1', '512-256', '64'], num_layers, emb_size, hidden_size, batchsize --> [:-1] batch size for translating not relevant
    #model_infos = experiment_path[0].split("/")[-1].split("_")[:-1]  #e.g. ['1-1', '512-256']

   # num_layers, sizes = model_infos #e.g. 1-1, 512-256
    #n_layers = num_layers[0]
    #emb_size = sizes[0]
    #hidden_size = sizes[1]
    checkpoints = []
    try:
        checkpoints = [f for f in os.listdir(os.path.join(start_root, experiment_path[0])) if
                   os.path.isfile(os.path.join(start_root, experiment_path[0], f)) and os.path.join(start_root, experiment_path[0], f).endswith("tar")]
    except FileNotFoundError or IndexError or IOError or RuntimeError as e:
        print("An error as occurred: %s" %e)
        print("Please run at least an experiment!")
        exit(-1)
    #print(checkpoints)
    checkpoints = sorted(checkpoints)
   # print(checkpoints)
    last_checkpoint = checkpoints[-1]
   # print(os.path.join(start_root, experiment_path[0], last_checkpoint))

    # Load model
    checkpoint = torch.load(os.path.join(start_root, experiment_path[0], last_checkpoint))
    # assert checkpoint

    enc = checkpoint['en']
    dec = checkpoint['de']
    src_vocab = checkpoint['src_dict']
    trg_vocab = checkpoint['tar_dict']
    src_emb = checkpoint['src_embedding']
    trg_emb = checkpoint['trg_embedding']
    enc_optimizer = checkpoint['en_opt']
    dec_optimzier = checkpoint['de_opt']
    layers = checkpoint['n_layers']
    hidden_size = checkpoint['hidden_size']

    src_voc = Voc("eng")
    trg_voc = Voc("deu")

    src_voc.__dict__ = src_vocab
    trg_voc.__dict__ = trg_vocab

    input_size = src_voc.num_words
    output_size = trg_voc.num_words

    emb_dim = src_emb['weight'].shape[1]

    src_embedding = nn.Embedding(input_size, emb_dim, _weight=src_emb['weight'])
    trg_embedding = nn.Embedding(output_size, emb_dim, _weight=trg_emb['weight'])

    encoder = EncoderLSTM(input_size, emb_size=emb_dim, hidden_size=hidden_size, n_layers=layers)
    decoder = DecoderLSTM(output_size, emb_size=emb_dim, hidden_size=hidden_size, n_layers=layers)

    encoder.load_state_dict(enc)
    decoder.load_state_dict(dec)

    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())

    encoder_optimizer.load_state_dict(enc_optimizer)
    decoder_optimizer.load_state_dict(dec_optimzier)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)

    evaluateInput(encoder, decoder, searcher, src_voc, trg_voc)


if __name__ == '__main__':
    ##### ArgumentParser ###########

    parser = argparse.ArgumentParser(description='PyTorch Vanilla LSTM Machine Translator')

    parser.add_argument('--path', type=str, default="",
                        help='experiment path')
    args = parser.parse_args()
    translate(".", path=args.path if args.path !="" else None)
