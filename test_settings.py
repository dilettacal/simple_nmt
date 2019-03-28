import os
import random

from global_settings import DATA_DIR, FILENAME
from utils.prepro import read_lines, preprocess_pipeline
from utils.tokenize import build_vocab, random_batch, batch2TrainData
from utils.utils import split_data


if __name__ == '__main__':
    start_root ="."
    expand_contraction = True
    pairs = read_lines(os.path.join(start_root, DATA_DIR), FILENAME)
    pairs, path = preprocess_pipeline(pairs, "cleaned.txt", expand_contraction, max_len=5)

    print(pairs[:10])

    src_sents = [item[0] for item in pairs]
    trg_sents = [item[1] for item in pairs]

    train_set, val_set, test_set = split_data(pairs, seed=1)

    input_lang = build_vocab(src_sents, "eng")
    output_lang = build_vocab(trg_sents, "deu")

    print("New batching solution:")
    seed = 10
    inp_var, inp_lengths, trg_var, trg_len, mask = random_batch(5, train_set, input_lang, output_lang, seed=10)

    print(trg_var)
    print(mask)

    print("Old batching solution:")
    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(input_lang, output_lang, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len, trg_len = batches

    print("input_variable:", input_variable)
    #print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
   # print("max_target_len:", max_target_len)