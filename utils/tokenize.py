# Default word tokens
import itertools
import random

import torch

"""
Mostly inspired by and adapted from: 
PyTorch Chatbot Tutorial: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
"""

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3 # Unknown token, no Key Error thrown

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {PAD: PAD_token, SOS: SOS_token, EOS: EOS_token, UNK: UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: PAD, SOS_token: SOS, EOS_token: EOS, UNK_token:UNK}
        self.num_words = 4  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def tensorize_sequence_list(self, seq_list, append_sos=False, append_eos = False):
        sos = SOS+" " if append_sos else ""
        eos = " "+ EOS if append_eos else ""
        seq_list = [sos + seq + eos for seq in seq_list if (append_sos and append_eos)]
        tensor = [[self.word2index[s] for s in sent.split(" ")] for sent in seq_list]
        return tensor

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('Words kept for {} vocabulary: {} / {} = {:.4f}'.format(self.name,
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {PAD: PAD_token, SOS: SOS_token, EOS: EOS_token, UNK: UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: PAD, SOS_token: SOS, EOS_token: EOS, UNK_token: UNK}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)

def build_vocab(sent_list, lang_name):
    vocab = Voc(name=lang_name)
    for sent in sent_list:
        vocab.addSentence(sent)
    return vocab


#### Trimming dictioanry if they are too big

def trimRareWords(voc, pairs, MIN_COUNT):
    #https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


###### Vectorization methods #######

def indexesFromSentence(voc, sentence):
    return [voc.word2index.get(word, 3) for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor(sorted([len(indexes) for indexes in indexes_batch], reverse=True))# 'lengths' array has to be sorted in decreasing order -> pack padded
    max_target_len = max(lengths)
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len, lengths

# Returns all items for a given batch of pairs
#### Batching from Chatbot tutorial #####
def batch2TrainData(src_voc, tar_voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, src_voc)
    output, mask, max_target_len, out_lengths = outputVar(output_batch, tar_voc)
    return inp, lengths, output, mask, max_target_len, out_lengths


##### Batching from practical pytorch ####

# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size, pairs, input_lang, output_lang, seed=1):
    random.seed(seed)
    input_seqs = []
    target_seqs = []
    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        #pair.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_seqs.append(indexesFromSentence(input_lang, pair[0]))
        target_seqs.append(indexesFromSentence(output_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]
    mask = binaryMatrix(target_padded)
    mask = torch.ByteTensor(mask).transpose(0,1)

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_padded).transpose(0, 1)
    target_var = torch.LongTensor(target_padded).transpose(0, 1)

    return input_var, input_lengths, target_var, target_lengths, mask