import itertools

import torch

PAD_token = "<PAD>"
SOS_token = "<SOS>"
EOS_token = "<EOS>"
UNK_token = "<UNK>"

PAD_idx = 0
SOS_idx = 1
EOS_idx = 2
UNK_idx = 3

#### Mapping token - indixes #####

class Vocab:
    def __init__(self, lang):
        self.lang = lang
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_idx: PAD_token, SOS_idx: SOS_token, EOS_idx: EOS_token, UNK_idx: UNK_token}
        self.num_words = len(self.index2word.keys()) # Count default tokens

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_token(word)

    def add_token(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_idx: PAD_token, SOS_idx: SOS_token, EOS_idx: EOS_token, UNK_idx: UNK_token}

        self.num_words = len(self.index2word.keys()) # Count default tokens

        for word in keep_words:
            self.add_token(word)

    def __repr__(self):
        return "%s with %s unique tokens" %(self.lang, len(self.index2word.keys()))


    @staticmethod
    def build_vocab_from_pairs(sentence_list, lang):
        voc = Vocab(lang=lang)
        for sent in sentence_list:
            if isinstance(sent, list):
                print("Sentence is passed as list...", sent)
                sentence = ' '.join([word for word in sent])
                voc.add_sentence(sentence)
            else:
                voc.add_sentence(sent)
        return voc


#### Vectorization methods #####


def indexesFromSentence(voc, sentence):
    return [voc.word2index.get(word, 3) for word in sentence.split(' ')] + [EOS_idx]


def zeroPadding(l, fillvalue=PAD_idx):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

#TO use in combination with CrossEntropyLoss and ignore_index=0
def seq2paddedTensor(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    max_len = max([len(indexes) for indexes in indexes_batch])
    return padVar, lengths, max_len


# Returns all items for a given batch of pairs
def batch2TrainData(src_voc, tar_voc, pair_batch):
    #print(pair_batch[0])
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths, _ = seq2paddedTensor(input_batch, src_voc)
    output, out_lengths, max_tar_len = seq2paddedTensor(output_batch, tar_voc)
    return inp, lengths, output, out_lengths, max_tar_len


### FUnctionalities used in tutorial ####
# Returns padded input sequence tensor and lengths
def inputVarTutorial(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_idx:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded target sequence tensor, padding mask, and max target length
def outputVarTutorial(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainDataTutorial(src_voc, tar_voc, pair_batch):
    #print(pair_batch[0])
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVarTutorial(input_batch, src_voc)
    output, mask, max_target_len = outputVarTutorial(output_batch, tar_voc)
    return inp, lengths, output, mask, max_target_len


