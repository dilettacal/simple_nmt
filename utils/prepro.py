import io
import os
import string
import unicodedata
import re
from pickle import dump, load

from utils.mappings import UMLAUT_MAP, ENG_CONTRACTIONS_MAP
from utils.tokenize import SOS_token, EOS_token
from global_settings import PREPRO_DIR

""" 
This script contains methods to preprocess text files.
Mostly borrowed and then adapted to this task from: 
- https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/
- https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/

Adaptation for this task:
- Tokens not simply reduced to their ASCII representation, but expanded according to local rules (see 'mappings.py')
    - If no local rules are available, the ASCII reduction as normalization method is applied

"""

def read_lines(root, filename):
    """
    Read line given the file root and the file name
    :param root: directory path
    :param filename: filename
    :return: Array of lines (parallel corpus)
    """
    path = os.path.join(root, filename)
    try:
        with io.open(path, encoding="utf-8", closefd=True) as f:
            lines = f.readlines()
    except IOError or FileNotFoundError or RuntimeError:
        print("Data file not found. Please run the script download.sh or download file manually from https://www.manythings.org/anki/, then extract files and copy them to ./data/")
    lines = [line.replace("\n", "").lower().split("\t") for line in lines]
    return lines

def reverse_language_pair(pairs):
    """
    To use after read_lines, this allows to reverse the language combination order
    :param pairs:
    :return:
    """
    return [list(reversed(p)) for p in pairs]


def unicode_to_ascii(s):
    """
    Convert the string s to ascii while removing accents
    'Mn' = mark non-spacing automatically removes accents, e.g. Umlaute
    This is a more rapid cleaning, the meaning of the sentence can be compromised
    (e.g. in German this removes the conditional form of werden and converts it to
    the past simple 'wurde(n)'
    :param s:
    :return: cleaned s

    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def preprocess_sentence(sentence, expand_contractions=None, append_token=False):
    #print("Preprocessing sentence...")
    """
    Rapid preprocessing
    https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/
    Pipeline:
    - Expand contractions if needed
    - Apply some filters, e.g. filter out sentences starting with certain prefixes
    - Remove non printable chars
    - Remove punctuation
    - Normalize chars to latin chars (e.g. accents simply removed)
    - Splitting on white spaces (== tokenizing)
    - Lower case
    - Remove digits
    :param lines: lines to be preprocessed
    :return: cleaned lines
    """
    if expand_contractions:
        mapping = expand_contractions
        sentence = expand_contraction(sentence, mapping)

    #Filtering function only applies on a sentence level
    #if filter_func:
        #sentence = filter_func(sentence)

    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    line = unicodedata.normalize("NFD", sentence).encode("ascii", "ignore")
    line = line.decode('UTF-8')
    # tokenize on white space
    line = line.strip().split(" ")
    # convert to lower case
    line = [word.lower() for word in line]
    # remove punctuation from each token
    line = [word.translate(table) for word in line]
    # remove non-printable chars form each token
    line = [re_print.sub('', w) for w in line]
    # remove tokens with numbers in them
    line = [word for word in line if word.isalpha()]

    line = ' '.join([word for word in line])

   # print(line)

    if append_token:
        line = [SOS_token] + line + [EOS_token]
    return line



def expand_contraction(sentence, mapping):
    """
    Expands tokens in sentence given a contraction dictionary

    source: https://www.linkedin.com/pulse/processing-normalizing-text-data-saurav-ghosh

    :param sentence: sentence to expand
    :param mapping: contraction dictionary
    :return: expanded sentence
    """
    contractions_patterns = re.compile('({})'.format('|'.join(mapping.keys())), flags=re.IGNORECASE | re.DOTALL)

    def replace_text(t):
        txt = t.group(0)
        if txt.lower() in mapping.keys():
            return mapping[txt.lower()]

    expanded_sentence = contractions_patterns.sub(replace_text, sentence)
    return expanded_sentence


def normalize_string(sentence):
    """
    Inspired by: PyTorch Tutorials
    Used in combination to @unicode_to_ascii function
    :param sentence:
    :param replace_dict:
    :return:
    """
    s = sentence
    s = re.sub(r"([ß])", r"ss", s)  # unicode_to_ascii cannot handle ß

    s = unicode_to_ascii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)

    return s


def reverse_order(token_list, reverse=True):
    """
    This apply order reversing to a list of sequences, e.g. the source sentences.
    :param token_list:
    :param reverse: Flag
    :return: reversed token list or token list if filter is not to apply
    """
    if reverse:
        token_list = token_list[::-1]
    return token_list


def save_clean_data(path, pairs, filename):
    """
    Saves the cleaned data as pickle file
    :param path:
    :param pairs:
    :param filename:
    :return:
    """
    path_to_dir = os.path.join(path, filename)
    dump(pairs, open(path_to_dir, 'wb'))
    print('File %s saved in %s' % (filename, path_to_dir))
    return path_to_dir

def load_cleaned_data(path, filename):
    """
    Loads pickle of cleaned files
    :param path:
    :param filename:
    :return:
    """
    path_to_file = os.path.join(path, filename)
    if(os.path.isfile(path_to_file)):
        return load(open(path_to_file, 'rb'))
    else: raise RuntimeError("File not found, please preprocess and save sentences!")


def preprocess_pipeline(pairs, cleaned_file_to_store=None, exp_contraction=None, reverse_pairs=False):
    """
    Performs preprocessing in a single pipeline
    :param cleaned_file_to_store:
    :param exp_contraction:
    :param reverse_pairs:
    :return:
    """

   # pairs = read_lines(DATA_DIR, FILENAME)
  #  print(len(pairs))
    #print(pairs[10])
    src_sents = [item[0] for item in pairs]
    trg_sents = [item[1] for item in pairs]
    if expand_contraction:
        src_mapping = ENG_CONTRACTIONS_MAP
        trg_mapping = UMLAUT_MAP
    else:
        src_mapping = exp_contraction
        trg_mapping = exp_contraction

    cleaned_pairs = []
    for i, (src_sent, trg_sent) in enumerate(pairs):
        cleaned_src_sent = preprocess_sentence(src_sent, src_mapping)
        cleaned_trg_sent = preprocess_sentence(trg_sent, trg_mapping)
        cleaned_list = [cleaned_src_sent, cleaned_trg_sent]

        cleaned_pairs.append(cleaned_list)

    if reverse_pairs:
        cleaned_pairs = reverse_language_pair(cleaned_pairs)

    if cleaned_file_to_store:
        store_path = save_clean_data(PREPRO_DIR, cleaned_pairs, cleaned_file_to_store)
        print("Preprocessing complete!")
    return cleaned_pairs, store_path