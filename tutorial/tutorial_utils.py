import torch

from data.tokenize import indexesFromSentence, zeroPadding, PAD_token, PAD_idx


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


def outputVarTutorial(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


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