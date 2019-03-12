import os
import random
import time
import math
import torch
from global_settings import SAVE_DIR, device
from data.tokenize import batch2TrainData

path_to_root = ".."


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_batches, optimizer, criterion, clip, teacher_force_ratio):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(train_batches):
        """
        Due to collate_fn each batch returns src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens
        For example:
        (('we did not get far',), 
        
        ('wir sind nicht sehr weit gekommen',), 
        
        tensor([[175],
        [115],
        [ 97],
        [ 52],
        [266],
        [  2]]), 
        
        tensor([[ 165],
        [ 351],
        [  69],
        [1180],
        [ 503],
        [ 327],
        [   2]]), [6], [7])
        
        """
        src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch

        #Debug information
        #src shape [max_len, batch_size]
        #trg_shape [max_len, batch_size]
        #torch.Size([13, 24])
        #torch.Size([15, 24])

        src_input_seqs = src_seqs.to(device)
        trg_output_seqs = tgt_seqs.to(device)
        optimizer.zero_grad()

        output = model(src_input_seqs, trg_output_seqs, teacher_force_ratio,src_lens)
        output = output.to(device)
        # output shape is seq_len, batch_size, output dim, e.g. torch.Size([13, 24, 32632])

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]
        # output.shape[-1] - last value (output dim)

        output = output.view(-1, output.shape[-1])
        # reshaped output [seq_len*batch_size, output_dim]

        trg_output_seqs = trg_output_seqs.view(-1)
        #reshaped for use in criterion: (max_len*batch_size) e.g. torch.Size([360])

        #CrossEntropyLoss criterion expects as an
        # input the score of each class (minibatch,C)
        # and the real target values as a tensor (N)

        loss = criterion(output, trg_output_seqs)

        #Compute gradients
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_batches)



def evaluate(model, val_batches, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(val_batches):
            src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch

            src_input_seqs = src_seqs.to(device)
            trg_output_seqs = tgt_seqs.to(device)

            output = model(src_input_seqs, trg_output_seqs, 0)  #while evaluating teacher forcing is turned off

            output = output.view(-1, output.shape[-1])
            trg_output_seqs = trg_output_seqs.view(-1)

            loss = criterion(output, trg_output_seqs)

            epoch_loss += loss.item()

    return epoch_loss / len(val_batches)


def evaluate_input(input):
    pass


def run_experiment(model, optimizer, num_epochs,criterion, clip, train_iter, val_iter, teacher_forcing_ratio=0.3):

    best_valid_loss = -1

    save_dir = os.path.join(path_to_root, SAVE_DIR)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        start_time = time.time()
        print("Computing train loss...")
        train_loss = train(model, train_iter, optimizer, criterion, clip, teacher_force_ratio=teacher_forcing_ratio)
        print("Computing validation loss....")
        valid_loss = evaluate(model, val_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_dir)
            print("Model saved!")

        print(f'| Epoch: {epoch + 1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} |  Val. Loss: {valid_loss:.3f} |')



