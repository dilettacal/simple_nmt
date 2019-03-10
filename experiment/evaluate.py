
import torch
import torch.nn as nn

from data.prepro import preprocess_sentence
from global_settings import device, MAX_LENGTH
from data.tokenize import SOS_idx, indexesFromSentence, EOS_token, SOS_token, PAD_token


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_idx
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores



def evaluate(encoder, decoder, searcher, src_voc, trg_voc, sentence, max_length=MAX_LENGTH, eval_input=True):
    if eval_input:
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
    else:
        #TODO: implement this!
        with torch.no_grad():
            # words -> indexes
            indexes_batch = [indexesFromSentence(src_voc, sentence)]
            # Create lengths tensor
            lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
            # Transpose dimensions of batch to match models' expectations
            input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
            # Use appropriate device
            input_batch = input_batch.to(device)
            lengths = lengths.to(device)


def evaluateInput(encoder, decoder, searcher,  src_voc, trg_voc, expand_contraction=None):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = preprocess_sentence(input_sentence, expand_contractions=expand_contraction)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, src_voc, trg_voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == EOS_token or x == PAD_token)]
            print('Translation:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")