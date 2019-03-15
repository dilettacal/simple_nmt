import itertools
import random

import torch
from torch.utils.data import Dataset, SubsetRandomSampler, Subset
import numpy as np

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


class Tokenizer(object):
    def __init__(self, cleaned_sentence_list, lang_name, is_source=False):
        self.vocab = Vocab.build_vocab_from_pairs(cleaned_sentence_list, lang=lang_name)
        self.is_source = is_source

    def index_from_sentences(self, sentence, append_SOS=True, append_EOS =True):
        #SOS for source sentences is meaningful when sentence is reversed or if we use a bidirectional cell
        idx = []
        if append_SOS:
            idx.append(SOS_idx)
        idx.extend(self.vocab.word2index.get(token, UNK_idx) for token in sentence.split(" "))
        if append_EOS:
            idx.append(EOS_idx)
        return idx

    def sentence_from_idx(self, idx_list):
        sent = [self.vocab.index2word.get(idx, UNK_token) for idx in idx_list]
        return sent, ' '.join([word for word in sent])

class NMTDataset(Dataset):
    #Inspired by: https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb

    def __init__(self, cleaned_pairs, src_lang, trg_lang,
                 src_tokenizer:Tokenizer, trg_tokenizer:Tokenizer):
        super(NMTDataset, self).__init__()
        self.pairs = cleaned_pairs
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_sents = [item[0] for item in self.pairs]
        self.trg_sents = [item[1] for item in self.pairs]
        self.src_reversed = False

    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        src_sent = self.src_sents[idx]
        trg_sent = self.trg_sents[idx]
        sent2tensor_src = self.src_tokenizer.index_from_sentences(src_sent, False, True)
        sent2tensor_trg = self.trg_tokenizer.index_from_sentences(trg_sent, False, True)
        return src_sent, trg_sent, sent2tensor_src, sent2tensor_trg

    def reverse_src_sents(self):
        self.src_sents = [sent[::-1] for sent in self.src_sents]
        self.src_reversed = True

    def reverse_language_pairs(self):
        temp = self.src_tokenizer
        self.src_tokenizer = self.trg_tokenizer
        self.trg_tokenizer = temp
        self.src_tokenizer.is_source = True
        self.trg_tokenizer.is_source = False

    def set_split(self, name, sampler:SubsetRandomSampler):
        idx = sampler.indices
        if name=="train":
            self.train = NMTDataset(Subset(self.pairs, idx), self.src_lang, self.trg_lang, self.src_tokenizer, self.trg_tokenizer)
        elif name == "val":
            self.val = NMTDataset(Subset(self.pairs, idx), self.src_lang, self.trg_lang, self.src_tokenizer, self.trg_tokenizer)
        elif name =="test":
            self.test = NMTDataset(Subset(self.pairs, idx), self.src_lang, self.trg_lang, self.src_tokenizer, self.trg_tokenizer)

    def get_overview(self, sub_set, quantity=5):
        overview = []
        for i in range(quantity):
            overview.append(random.choice(sub_set))
        return overview



#### Vectorization methods #####
### TODO: Remove if system works, as they are deprecated

def indexesFromSentence(voc, sentence):
    return [SOS_idx] + [voc.word2index.get(word, UNK_idx) for word in sentence.split(' ')] + [EOS_idx]


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


def collate_fn(data):

    ### Source: https://github.com/howardyclo/pytorch-seq2seq-example/blob/master/seq2seq.ipynb

    """
    Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
        - src_sents, tgt_sents: batch of original tokenized sentences
        - src_seqs, tgt_seqs: batch of original tokenized sentence ids
    Returns:
        - src_sents, tgt_sents (tuple): batch of original tokenized sentences
        - src_seqs, tgt_seqs (variable): (max_src_len, batch_size)
        - src_lens, tgt_lens (tensor): (batch_size)

    """

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        #Sorting lengths for pad_pack_sequence
        lens = sorted(lens, reverse=True)
        return padded_seqs, lens

    # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
    # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
    # *source* sequence, which is in the EncoderRNN
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences.
    src_sents, tgt_sents, src_seqs, tgt_seqs = zip(*data)

    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lens = _pad_sequences(src_seqs)
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seqs)

    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0, 1)
    tgt_seqs = tgt_seqs.transpose(0, 1)

    return src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens

