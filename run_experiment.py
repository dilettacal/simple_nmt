import argparse
import os
import random
from datetime import datetime
import torch
from torch import optim

from experiment.train_eval import evaluateInput, GreedySearchDecoder, trainIters, eval_batch, plot_training_results
from global_settings import device, FILENAME, SAVE_DIR, PREPRO_DIR, TRAIN_FILE, TEST_FILE, EXPERIMENT_DIR, LOG_FILE
from model.model import EncoderLSTM, DecoderLSTM
from utils.prepro import read_lines, preprocess_pipeline, load_cleaned_data, save_clean_data
from utils.tokenize import build_vocab, batch2TrainData

from global_settings import DATA_DIR
from utils.utils import split_data, filter_pairs, max_length, plot_grad_flow


def str2bool(v):
    #https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def str2float(s):
    try:
        return float(s)
    except ValueError:
        return None





if __name__ == '__main__':

    ##### ArgumentParser ###########

    parser = argparse.ArgumentParser(description='PyTorch Vanilla LSTM Machine Translator')
    #parser.add_argument('--data', type=str, default='./data/',
                      #  help='location of the data corpus. Default in ./data/')
    ### Embedding size ####
    parser.add_argument('--emb', type=int, default=256,
                        help='size of word embeddings')
    ### Hidden size ####
    parser.add_argument('--hid', type=int, default=256,
                        help='number of hidden units per layer')

    ### Number of layers ####
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')

    ### Learning rate ###
    parser.add_argument('--lr', type=float, default=0.003,
                        help='initial learning rate')

    ### Gradient clipping ###
    parser.add_argument('--clip', type=str2float, default="",
                        help='gradient clipping. Provided as a float number or an empty string \" \", if no clipping should happen.')

    ### Number of iterations ###
    parser.add_argument('--iterations', type=int, default=15000,
                        help='number of iterations')

    ### Batch size ###
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')

    ### Teacher forcing ratio ###
    parser.add_argument('--teacher', type=float, default=0.2, help="Teacher forcing ration during training phase")

    ### How many data ###
    parser.add_argument('--limit', type=int, help='Reduce dataset to N samples')

    ### Decoder learning rate ###
    parser.add_argument('--dec_lr', type=int, default=1, help="Decoder learning rate decay. This must be provided as integer, as it is multiplied by the learning rate (lr)")

    ### Compute vocabulary on all dataset or only training samples ###
    parser.add_argument('--voc_all', type=str2bool, nargs='?',
                        const=True, default="True",
                        help="Get vocabulary from all dataset (true) or only from training data (false).\n"
                             "Possible inputs: 'yes', 'true', 't', 'y', '1' OR 'no', 'false', 'f', 'n', '0'")

    ### Truncated Backprop through time ###
    parser.add_argument('--tbptt', type=str2bool, default="True",
                        help="Set how to perform truncation in backpropagation. If 'true', every time 'detach()' is applied on the hidden states. "
                             "If 'false', 'detach()' is not applied.\n"
                             "Possible inputs: 'yes', 'true', 't', 'y', '1' OR 'no', 'false', 'f', 'n', '0'")
    ### Dropout ###
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0.0 = no dropout). Values range allowed: [0.0 - 1.0]')

    ### Seed ###
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    ### Run program on cuda ###
    parser.add_argument('--cuda', type=str2bool, default="true", help="use CUDA.\n"
                                                                      "Possible inputs: 'yes', 'true', 't', 'y', '1' OR 'no', 'false', 'f', 'n', '0'")

    ### Logging interval ###
    parser.add_argument('--log_interval', type=int, default=100, help='report interval')

    parser.add_argument('--max_len', type=int, default=0, help='max sentence length in the dataset. Sentences longer than max_len are trimmed. Provide 0 for no trimming!')

    parser.add_argument('--cell', type=str, default="lstm", help="Cell type. Allowed values: 'lstm' (default)' or 'gru'")


    #### Start #####

    # Read arguments
    args = parser.parse_args()

    print("Expreiment settings:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")

    ###### Starting the program #####

    experiment_execution_time = datetime.now()

    start_root = "."
    src_lang = "eng"
    trg_lang = "deu"
    exp_contraction = True

    limit = args.limit

    voc_all = args.voc_all

    ### Setup preprocessing file ####

    max_sent_len = args.max_len
    if max_sent_len > 0:
        cleaned_file =  "%s-%s_cleaned" % (src_lang, trg_lang) + "_{}".format(max_sent_len) +".pkl"
    else:
        cleaned_file = "%s-%s_cleaned" % (src_lang, trg_lang) + "_full" + ".pkl"

    ### Check if data has already been preprocessed, if not, preprocess it ####

    if os.path.isfile(os.path.join(PREPRO_DIR, cleaned_file)):
        print("File already preprocessed! Loading file....")
        pairs = load_cleaned_data(PREPRO_DIR, filename=cleaned_file)
    else:
        print("No preprocessed file found. Starting data preprocessing...")
        pairs = read_lines(os.path.join(start_root, DATA_DIR), FILENAME)
        pairs, path = preprocess_pipeline(pairs, cleaned_file, exp_contraction, max_len = max_sent_len) #data/prepro/eng-deu_cleaned_full.pkl

    ### Get sample ###
    print("Sample from data:")
    print(random.choice(pairs))

    src_sents, trg_sents = [], []

    if voc_all:
        # build vocabularies based on all data set (test set included)
        src_sents = [item[0] for item in pairs]
        trg_sents = [item[1] for item in pairs]

    print("Limit set: %s" %str(limit))

    if limit:
        pairs = pairs[:limit]

    train_set, val_set, test_set = split_data(pairs, seed=args.seed)

    print("Data in train set:", len(train_set))
    print("Data in val set:", len(val_set))
    print("Data in test set:", len(test_set))


    print("Building vocabularies...")
    train_data = train_set+val_set


    if not voc_all:
        # Build vocabularies based on train set
        src_sents = [item[0] for item in train_data]
        trg_sents = [item[1] for item in train_data]

   # print("Source:", src_sents)
   # print("Target:", trg_sents)

    max_src_l = max_length(src_sents)
    max_trg_l = max_length(trg_sents)

    print("Max sentence length in source sentences:", max_src_l)
    print("Max sentence length in source sentences:", max_trg_l)

    input_lang = build_vocab(src_sents, "eng")
    output_lang = build_vocab(trg_sents, "deu")

    print("Source vocabulary:", input_lang.num_words)
    print("Target vocabulary:", output_lang.num_words)


    test_batches = [batch2TrainData(input_lang, output_lang, [test_set[i]])
                        for i in range(len(test_set))]


    print("Test batches:", len(test_batches))


    # Configure models
    hidden_size = args.hid
    encoder_n_layers = args.nlayers
    decoder_n_layers = args.nlayers
    batch_size = args.batch_size
    input_size = input_lang.num_words
    output_size = output_lang.num_words
    embedding_size = args.emb
    dropout = args.dropout
    tbptt = args.tbptt
    clip = args.clip
    teacher_forcing_ratio = args.teacher
    learning_rate = args.lr
    decoder_learning_ratio = args.dec_lr
    n_iteration = args.iterations
    val_iteration = n_iteration
    print_every = args.log_interval

    cell_type = args.cell
    if cell_type not in ["lstm", "gru"]:
        cell_type = "lstm"
        print("{} cell type not allowed. Cell type has been set to default value 'lstm'".format(args.cell))

    save_every = 500

    model_name = ''
    model_name += 'simple_nmt_model' + str(limit) if limit else 'simple_nmt_model_full_' + str(len(pairs))
    model_name += "_teacher_{}".format(str(teacher_forcing_ratio)) if teacher_forcing_ratio > 0.0 else "_no_teacher"
    model_name += "" if voc_all else "train_voc"

    print('Building encoder and decoder ...')
    encoder = EncoderLSTM(input_size=input_size, emb_size=embedding_size, hidden_size=hidden_size,
                         n_layers=encoder_n_layers, dropout=dropout, cell_type=cell_type)
    decoder = DecoderLSTM(output_size=output_size, emb_size=embedding_size, hidden_size=hidden_size,
                          n_layers= decoder_n_layers, dropout=dropout, cell_type=cell_type)

    assert encoder.cell_type == decoder.cell_type
    assert encoder.n_layers == decoder.n_layers

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built:')
    print(encoder)
    print(decoder)

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adamax(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adamax(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)


    # Run training iterations
    print("Starting Training!")
    start_time = datetime.now()
    val_loss, directory, train_history, val_statistics, _, _ = \
        trainIters(model_name, input_lang, output_lang, train_set, val_set, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   encoder_n_layers, decoder_n_layers, SAVE_DIR, n_iteration, batch_size,
                   print_every, save_every, clip, FILENAME, val_iteration, tbptt=tbptt)

    end_time = datetime.now()
    duration = end_time-start_time
    print('Training duration: {}'.format(duration))

    print("Performing evaluation on test set...")
    test_loss = eval_batch(test_batches, encoder, decoder)
    print("Test loss:", test_loss)

    print("Checkponits saved in %s" %(directory))

    #Log file name
    try:
        with open(os.path.join(start_root, EXPERIMENT_DIR, LOG_FILE), encoding="utf-8", mode="w") as f:
            #Logging to last_experiment.txt
            f.write(str(directory))
        with open(os.path.join(start_root, EXPERIMENT_DIR, "log_history.txt"), encoding="utf-8", mode="a") as hf:
            #Logging in the history document
            hf.write("Execution date: %s" % str(experiment_execution_time))
            hf.write("\nExperiment name:\n")
            hf.write(model_name)
            hf.write("\nDirectory:\n")
            hf.write(str(directory))
            hf.write("\n Number of samples: %s" %str(len(pairs)))
            if voc_all:
                hf.write("\nVocabularies built on all dataset")
            else:
                hf.write("\nVocabularies built only on the train set")
            hf.write("\nSource vocabulary: %s" % str(input_lang.num_words))
            hf.write("\nTarget vocabulary: %s" % str(output_lang.num_words))
            hf.write("\nMax src length %s" % max_src_l)
            hf.write("\nMax trg length %s" % max_trg_l)
            hf.write("\nLearning rate: %s" % str(learning_rate))
            hf.write("\nBatch size: %s" % str(batch_size))
            hf.write("\nEmbedding size: %s" % str(embedding_size))
            hf.write("\nHidden size: %s" % str(hidden_size))
            hf.write("\nAverage validation loss: %s" %str(val_loss))
            hf.write("\nTest loss: %s" %str(test_loss))
            hf.write("\nTraining iterations:")
            hf.write(str(n_iteration))
            hf.write("\n Training duration:")
            hf.write(str(duration))
            hf.write("\n**********************************\n")
    except IOError or TypeError or RuntimeError:
        print("Log to file failed!")

    print("Plotting results...")
    try:
        plot_training_results(model_name, train_history, val_statistics[0], SAVE_DIR, FILENAME, decoder_n_layers, embedding_size, hidden_size, batch_size, learning_rate,
                              n_iterations = n_iteration, log_interval = print_every, val_plot = val_statistics[1], live_show=False)
        print("Plots stored!")

    except IOError or RuntimeError:
        pass