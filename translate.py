"""
Model is saved:


torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': val_loss,
                'src_dict': src_voc.__dict__,
                'tar_dict': tar_voc.__dict__,
                'src_embedding': encoder.embedding.state_dict(),
                'trg_embedding': decoder.embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

Directory:
experiment/checkpoints/simple_nmt_model30000/deu.txt/1-1_256

"""
import os

import torch
from torch import nn, optim

from experiment.train_eval import GreedySearchDecoder, evaluateInput
from global_settings import EXPERIMENT_DIR, LOG_FILE, device
from model.model import EncoderLSTM, DecoderLSTM
from utils.tokenize import Voc

if __name__ == '__main__':

    start_root = "."

    print("Reading experiment information...")
    with open(os.path.join(start_root, EXPERIMENT_DIR, LOG_FILE), mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    experiment_path = [line.strip() for line in lines if "experiment/" in line]
  #  print(experiment_path[0])
    hidden_size = int(experiment_path[0].split("_")[-1])
   # print(hidden_size)
    checkpoints = [f for f in os.listdir(os.path.join(start_root, experiment_path[0])) if os.path.isfile(os.path.join(start_root, experiment_path[0], f))]
    last_checkpoint = checkpoints[-1]

    #Load model
    checkpoint = torch.load(os.path.join(start_root, experiment_path[0], last_checkpoint))
    assert checkpoint

    enc = checkpoint['en']
    dec= checkpoint['de']
    src_vocab = checkpoint['src_dict']
    trg_vocab = checkpoint['tar_dict']
    src_emb = checkpoint['src_embedding']
    trg_emb = checkpoint['trg_embedding']
    enc_optimizer = checkpoint['en_opt']
    dec_optimzier = checkpoint['de_opt']

    src_voc = Voc("eng")
    trg_voc = Voc("deu")

    src_voc.__dict__ = src_vocab
    trg_voc.__dict__ = trg_vocab

    input_size = src_voc.num_words
    output_size = trg_voc.num_words

    emb_dim = src_emb['weight'].shape[1]

    src_embedding = nn.Embedding(input_size, emb_dim, _weight=src_emb['weight'])
    trg_embedding = nn.Embedding(output_size, emb_dim, _weight=trg_emb['weight'])

    encoder = EncoderLSTM(input_size,emb_size=emb_dim,hidden_size=hidden_size)
    decoder = DecoderLSTM(output_size,emb_size=emb_dim, hidden_size=hidden_size)

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
