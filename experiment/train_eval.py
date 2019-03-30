import os
import random
from torch import nn
from global_settings import device, MAX_LENGTH, VAL_TRAIN_DELTA
import torch
from utils.prepro import preprocess_sentence
from utils.tokenize import SOS_token, batch2TrainData, indexesFromSentence, EOS, PAD, EOS_token
from utils.utils import maskNLLLoss
from global_settings import NUM_BAD_VALID_LOSS, LR_DECAY, MIN_LR, MAX_LR

## Truncated backpropagation
def detach_states(states):
    #https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py#L59
    #a lighter implementation of 'repackage_hidden' from: https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L103
	if states is None:
		return states
	return [state.detach() for state in states]

### Manage learning rate during training
### Manage optimizers: https://pytorch.org/docs/master/optim.html

def adapt_lr(enc_optim, dec_optim, decay_value):
    for param_group in enc_optim.param_groups:
        param_group['lr'] *= decay_value
    new_enc_lr = enc_optim.param_groups[0]['lr']
    print("New encoder optimizer learning rate {}".format(new_enc_lr))

    for param_group in dec_optim.param_groups:
        param_group['lr'] *= decay_value

    new_dec_lr = dec_optim.param_groups[0]['lr']
    print("New decoder optimizer learning rate {}".format(new_dec_lr))

    return new_enc_lr, new_dec_lr

def train(input_variable, lengths, target_variable, mask, max_target_len, trg_lengths, encoder, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio=0.5, K=5, tbptt=True):
    """
    Performs a training step on a batch during training process
    :param input_variable: batched tensor input
    :param lengths: lengths of input variable
    :param target_variable: batched tensor target
    :param mask: masking for this input and target variables
    :param max_target_len: maximum length in target
    :param trg_lengths: lengths of the target variables (actually not used)
    :param encoder: encoder
    :param decoder: decoder
    :param encoder_optimizer: encoder optimizer
    :param decoder_optimizer: decoder optimizer
    :param batch_size: batch size
    :param clip: gradient clipping
    :param teacher_forcing_ratio: frequency to use teacher forcing
    :return: the train loss
    """
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    #trg_lengths = trg_lengths.to(device) #RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

   # K = max_target_len//2

    # Forward pass through encoder
    encoder_outputs, encoder_states = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    #decoder_hidden = encoder_hidden[:decoder.n_layers]
    decoder_states = encoder_states


    # Determine if we are using teacher forcing this iteration
    rand = random.random()
    # print(rand)
    use_teacher_forcing = True if rand < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_states = decoder(
                decoder_input, decoder_states
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
            decoder_output, decoder_states = decoder(
                decoder_input, decoder_states
            )
            ### Truncate backpropagation through time ###
            if tbptt:
                #decoder_input = decoder_input.detach()
                decoder_states = detach_states(decoder_states)
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

    if clip:
        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def eval(input_variable, lengths, target_variable, mask, max_target_len, trg_lengths, encoder, decoder,
          batch_size):
    """
    Performs evaluation on validation set during training iteration
    :param input_variable: batched tensor input
    :param lengths: lengths of input variable
    :param target_variable: batched tensor target
    :param mask: masking for this input and target variables
    :param max_target_len: maximum length in target
    :param trg_lengths: lengths of the target variables (actually not used)
    :param encoder: encoder
    :param decoder: decoder
    :param batch_size: batch size
    :return: validation loss
    """
    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    with torch.no_grad():

        # Forward pass through encoder
        encoder_outputs, encoder_states = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        #decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_states = encoder_states

        # Forward batch of sequences through decoder one time step at a time
        for t in range(max_target_len):
            decoder_output, decoder_states = decoder(
                decoder_input, decoder_states
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = decoder_input.detach()
            try:
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]) #RuntimeError: CUDA error: device-side assert triggered
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
            except RuntimeError:
                pass


        return sum(print_losses) / n_totals

def trainIters(model_name, src_voc, tar_voc, train_pairs, val_pairs, encoder, decoder,
               encoder_optimizer, decoder_optimizer,
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every,
               save_every, clip, corpus_name, val_iterations, tbptt=True):
    """
    This method defines the main training procedure
    :param model_name: model name
    :param src_voc: source vocabulary
    :param tar_voc: target vocabulary
    :param train_pairs: train set
    :param val_pairs: validation set
    :param encoder: encoder object
    :param decoder: decoder object
    :param encoder_optimizer: encoder optimizer, by default: Adam
    :param decoder_optimizer: decoder optimizer, by default: Adam
    :param encoder_n_layers: encoder number of layers
    :param decoder_n_layers: decoder number of layers
    :param save_dir: store directory path
    :param n_iteration: number of iterations to be done
    :param batch_size: batch size
    :param print_every: frequency of printing results
    :param save_every: storage frequency
    :param clip: gradient clipping value
    :param corpus_name: file name
    :return: average validation loss, directory, train_history, val_history
    """
    # Load batches for each iteration
    global directory
    directory = ""

    best_validation_loss = float('inf')
    n_bad_loss=0

    random.seed(1)
    training_batches = [batch2TrainData(src_voc, tar_voc, [random.choice(train_pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    val_batches = [batch2TrainData(src_voc, tar_voc, [val_pairs[i]]) for i in range(len(val_pairs))]

    #### Directory setup

    directory = os.path.join(save_dir, model_name, corpus_name,
                             '{}-{}_{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, encoder.emb_size,
                                                     encoder.hidden_size, batch_size))
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print("An experiment with these settings has been already excuted. Deleting content...")
        files_to_remove = [os.path.join(directory, f) for f in os.listdir(directory)]
        for f in files_to_remove:
            if os.path.isfile(f):
                os.remove(f)

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    train_print_loss = 0
    val_print_loss = 0
    print_val_loss_avg = 0
    print_loss_avg = 0

    ##### Store plot results
    train_history = []
    val_history = []

    encoder_avg_grads, decoder_avg_grads = [], []
    encoder_layers, decoder_layers = [], []

    for iteration in range(start_iteration, n_iteration):
        leave_training = False
        # Get the actual batch
        encoder.train()
        decoder.train()

        training_batch = training_batches[iteration - 1]
        train_inp_var, train_src_len, train_trg_var, train_mask, train_max_len, train_trg_len = training_batch
        train_loss = train(train_inp_var, train_src_len, train_trg_var, train_mask, train_max_len, train_trg_len,
                           encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, clip, K=0, tbptt=tbptt)



        train_print_loss += train_loss

         #### store results
        train_history.append(train_loss)

        encoder.eval()
        decoder.eval()


        layers = encoder.n_layers
        hidden_size = encoder.hidden_size
        val_loss = 0

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = train_print_loss / print_every
            val_loss = eval_batch(val_batches, encoder, decoder)

            #print_val_loss_avg = val_loss / print_every
            print_val_loss_avg =val_loss
            print("Iteration: {}; Percent complete: {:.1f}%; Average train loss: {:.4f}; Average val loss: {:.4f}"
                  .format(iteration, iteration / n_iteration * 100, print_loss_avg, print_val_loss_avg))
            train_print_loss = 0
            val_print_loss = 0
            print(val_loss - train_loss)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                #### Saving the model....
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
                    'trg_embedding': decoder.embedding.state_dict(),
                    'n_layers': layers,  # Layer numbers the same for both components
                    'hidden_size': hidden_size

                }, os.path.join(directory, '{}.tar'.format('checkpoint')))
            else:
                n_bad_loss +=1
            if n_bad_loss == NUM_BAD_VALID_LOSS or (val_loss - train_loss > VAL_TRAIN_DELTA):
                n_bad_loss = 0
                new_lr_enc, new_lr_dec = adapt_lr(encoder_optimizer, decoder_optimizer, LR_DECAY)

                if new_lr_enc < MIN_LR and new_lr_dec < MIN_LR:
                    leave_training = True
                    break

        val_history.append(val_loss)

        if leave_training:
            print("Stopping training...")
            break


    return print_val_loss_avg, directory, train_history, val_history, [encoder_avg_grads, encoder_layers], [decoder_avg_grads, decoder_layers]



class GreedySearchDecoder(nn.Module):
    """
    This is a greedy searcher decoder, to use during inference
    """
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        #decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_hidden = encoder_hidden
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


############# Evaluation ################

def evaluate(searcher, src_voc, trg_voc, sentence, max_length=MAX_LENGTH):
    """
    Util method to do evaluation
    :param searcher: the searcher method. By Default it is a GreedySearcher
    :param src_voc: the source vocabulary
    :param trg_voc: the target vocabulary
    :param sentence: the sentence to be translated
    :param max_length: search max length
    :return: Decoded words
    """
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(src_voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [trg_voc.index2word[token.item()] for token in tokens]
    return decoded_words


######## Inference from input ##############

def evaluateInput(encoder, decoder, searcher,  src_voc, trg_voc, from_file = None):
    """
    Adapted from: PyTorch Chatbot Tutorial
    Reads a sentence as keyboard input and returns its translation
    :param encoder:
    :param decoder:
    :param searcher:
    :param src_voc:
    :param trg_voc:
    :param from_file: the file which sentences are read from
    :return:
    """
    results = []

    if from_file:
        for line in from_file:
            line = preprocess_sentence(line)
            output_words = evaluate(searcher, src_voc, trg_voc, line)
            output_words[:] = [x for x in output_words if not (x == EOS or x == PAD)]
            if output_words:
                translation = ' '.join(output_words)
                results.append([line, str(translation)])
            else:
                results.append([line, "No translation"])
        return results
    else:
        input_sentence = ''
        while(1):
            try:
                # Get input sentence
                input_sentence = input('Source > ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                # Normalize sentence
                input_sentence = preprocess_sentence(input_sentence)
                # Evaluate sentence
                output_words = evaluate(searcher, src_voc, trg_voc, input_sentence)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == EOS or x == PAD)]
                if output_words:
                    print('Translation > ', ' '.join(output_words))
                else:
                    print("No translation found!")

            except KeyError:
                print("Error: Encountered unknown word.")


#### Evaluation on test set

def eval_batch(batch_list, encoder, decoder):
    """
    Performs evaluation on test set
    :param batch_list:
    :param encoder:
    :param decoder:
    :return:
    """
    total_loss = 0
    for batch in batch_list:
        test_inp_var, test_src_len, test_trg_var, test_mask, test_max_len, test_trg_len = batch

        test_loss = eval(test_inp_var, test_src_len, test_trg_var, test_mask, test_max_len, test_trg_len, encoder, decoder,
                    1)
        total_loss+= test_loss
    return total_loss/len(batch_list)

#### Plot results

def plot_training_results(modelname, train_history, val_history, save_dir, corpus_name, n_layers, embedding_size, hidden_size, bs, lr, live_show=False):
    """
    Plots training results
    :param modelname:
    :param train_history:
    :param val_history:
    :param save_dir:
    :param corpus_name:
    :param n_layers:
    :param hidden_size:
    :param live_show:
    :return: None, it stores the files or shows them
    """
    import matplotlib.pyplot as plt


    directory = os.path.join(save_dir, modelname, corpus_name,
                             '{}-{}_{}-{}_{}'.format(n_layers, n_layers, embedding_size, hidden_size, bs))

    directory = os.path.join(directory, "plots")

    if not os.path.isdir(directory):
        os.makedirs(directory)

    plt.plot(train_history)
    plt.plot(val_history)
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('iteration - lr= {}'.format(lr))
    plt.legend(['train', 'validation'], loc='upper right')
    if live_show: plt.show()
    file = "train_loss.png"
    path_to_file = os.path.join(directory, file)
    plt.savefig(path_to_file)
    plt.close()
