# Default word tokens
import itertools
import torch

"""
Mostly taken and adapted from: 
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


def build_vocab(sent_list, lang_name):
    vocab = Voc(name=lang_name)
    for sent in sent_list:
        vocab.addSentence(sent)
    return vocab


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
def batch2TrainData(src_voc, tar_voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, src_voc)
    output, mask, max_target_len, out_lengths = outputVar(output_batch, tar_voc)
    return inp, lengths, output, mask, max_target_len, out_lengths




