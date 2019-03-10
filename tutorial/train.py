import os
import random

import torch
from torch.utils import checkpoint

from data.utils import maskNLLLoss
from global_settings import  device
from data.tokenize import SOS_idx, batch2TrainDataTutorial


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
         encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_idx for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    rand = random.random()
    # print(rand)
    use_teacher_forcing = True if rand < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def evaluate(val_in_variable, in_variable_len, val_tar_variable, tar_mask, max_target_len, encoder, decoder, batch_size):

    # Set device options
    input_variable = val_in_variable.to(device)
    lengths = in_variable_len.to(device)
    target_variable = val_tar_variable.to(device)
    mask = tar_mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_idx for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]


    with torch.no_grad():

        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

        return sum(print_losses) / n_totals


def run_experiment(model_name, src_voc, tar_voc, encoder, decoder,
                   encoder_optimizer, decoder_optimizer,
                   src_embedding, trg_embedding,
                   encoder_n_layers, decoder_n_layers,
                   save_dir, n_iteration, batch_size, print_every,
                   save_every, clip, corpus_name, loadFilename, hidden_size, train_set_pairs, val_set_pairs, teacher_forcing_ratio=0):


    # Load batches for each iteration
    training_batches = [batch2TrainDataTutorial(src_voc, tar_voc, [random.choice(train_set_pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Load batches for each iteration
    val_batches = [batch2TrainDataTutorial(src_voc, tar_voc, [random.choice(val_set_pairs) for _ in range(batch_size)])
                   for _ in range(n_iteration)]


    # keeps track of the best valid loss
    best_valid_loss = float('inf')

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    train_print_loss = 0
    val_print_loss = 0

    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    for iteration in range(start_iteration, n_iteration + 1):

        ##################  Training ##########################

        # train mode
        encoder.train()
        decoder.train()

        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio=teacher_forcing_ratio)
        train_print_loss += loss


        ##################  Validation  ##########################

        validation_bacth = val_batches[iteration-1]
        val_in_variable, in_variable_len, val_tar_variable, tar_mask, tar_variable_len = validation_bacth

        encoder.eval()
        decoder.eval()

        val_loss = evaluate(val_in_variable, in_variable_len, val_tar_variable, tar_mask, tar_variable_len,
                        encoder, decoder, batch_size)

        val_print_loss += val_loss

        #print("Train loss on train set:", loss)
        #print("Val loss on val set:", val_loss)

        print_loss_avg = train_print_loss / print_every
        print_val_loss_avg = val_print_loss / print_every
        #print_val_loss_avg = val_print_loss
        # Print progress

        if iteration % print_every == 0:
            print("Iteration: {}; Percent complete: {:.1f}%; Average train loss: {:.4f}, Average val loss: {:.4f}".format(iteration,
                                                                                              iteration / n_iteration * 100,
                                                                                              print_loss_avg, print_val_loss_avg))

            train_print_loss = 0
            val_print_loss = 0


        if val_loss < best_valid_loss:
            best_valid_loss = print_val_loss_avg

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': val_loss,
                'src_dict': src_voc.__dict__,
                'tar_dict': tar_voc.__dict__,
                'src_embedding': src_embedding.state_dict(),
                'trg_embedding': trg_embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))




