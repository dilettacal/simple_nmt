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

#1 batch is composed of: inp, lengths, output, out_lengths, max_tar_len
# inp and output are the variable arrays

def train(model, train_batches, optimizer, criterion, clip, teacher_force_ratio):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(train_batches):

        src, src_lengths, trg, trg_lengths, trg_max_len = batch
        #Debug information
        #src shape [max_len, batch_size]
        #trg_shape [max_len, batch_size]
        start_src = src.shape
        start_trg = trg.shape

        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg, teacher_force_ratio,src_lengths)
        output = output.to(device)
        raw_o = output

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        # reshaped output [seq_len*batch_size, output_dim]

       # print("Target at index 0:", trg[0:]) #SOS token

        trg = trg[1:].view(-1) #not include sos token

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

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

            src, src_lengths, trg, trg_lengths, trg_max_len = batch
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(val_batches)


def evaluate_input(input):
    pass


def run_experiment(src_voc, tar_voc, model, optimizer, num_epochs,train_iteration, val_iteration,criterion, clip,
                   train_set, eval_set, train_batch_size,
                   val_batch_size, teacher_forcing_ratio=0.3):

    best_valid_loss = -1

    # Load batches for each iteration
    training_batches = [batch2TrainData(src_voc, tar_voc, [random.choice(train_set) for _ in range(train_batch_size)])
                        for _ in range(train_iteration)]
    #print(len(training_batches))

    # Load batches for each iteration
    val_batches = [batch2TrainData(src_voc, tar_voc, [random.choice(eval_set) for _ in range(val_batch_size)])
                   for _ in range(val_iteration)]

    save_dir = os.path.join(path_to_root, SAVE_DIR)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        start_time = time.time()
        print("Computing train loss...")
        train_loss = train(model, training_batches, optimizer, criterion, clip, teacher_force_ratio=teacher_forcing_ratio)
        print("Computing validation loss....")
        valid_loss = evaluate(model, val_batches, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_dir)

        print(f'| Epoch: {epoch + 1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')



