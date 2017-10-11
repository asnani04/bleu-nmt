import numpy as np
import codecs
import re
import itertools
from collections import Counter, OrderedDict


data_size = 150000 # size of the dataset required

def get_sentences(filename):
    '''
    Get sentences from the file in a list
    '''
    desc = codecs.open(filename, "rb", "utf-8")
    sents = []
    for line in desc.readlines()[:data_size]:
        line = line.strip().lower()
        sents.append(line)
    return sents

def tokenize(sents):
    ''' Tokenize the sentences '''
    tokenized = []
    for sent in sents:
        tokenized.append(sent.split(" "))
    return tokenized

def build_dataset(orig, vocabulary_size):
    ''' Build a vocabulary of words '''
    count = [['<unk>', -1]]
    sentences = []
    for sentence in orig:
        sentences.append(sentence)
    count.extend(Counter(itertools.chain(*sentences))
                 .most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data_orig = list()
    unk_count = 0
    for ind, sentence in enumerate(orig):
        data_orig.append([])
        for word in sentence:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data_orig[ind].append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data_orig, count, dictionary, reverse_dictionary

def create_datafile(sents, filename):
    ''' Create a file with each element of sents on a separate line.
    Args:
    @sents: list of sents to be put in file
    @filename: name of the file created

    returns: Number of sentences processed
    '''
    new_file = codecs.open(filename, "wb", "utf-8")
    for sent in sents:
        new_file.write(sent + "\n")
    return len(sents)

def create_vocab_file(rev_dict, filename):
    ''' Create a file with each word in the vocab on a separate line.'''
    new_file = codecs.open(filename, "wb", "utf-8")
    word_count = 0
    # The below line was throwing up error with unk's vocab code for some reason.
    # new_file.write("<unk>\n<s>\n</s>\n")
    for i, ind in enumerate(rev_dict.keys()):
        if i > 0 and rev_dict[i] != '':
            new_file.write(rev_dict[ind] + "\n")
            word_count += 1
    return word_count

def process(dir_name, lang1):
    vocab_size = 10000
    filename = "news-commentary-v11." + lang1 + "-en." + lang1
    filename = dir_name + "/" + filename
    orig = get_sentences(filename)
    filename = dir_name + "/" + "news-commentary-v11." + lang1 + "-en.en"
    altered = get_sentences(filename)

    orig_tokenized = tokenize(orig)
    altered_tokenized = tokenize(altered)
    
    data_orig, count_orig, orig_dict, orig_rev_dict = build_dataset(
        orig_tokenized, vocab_size)
    data_alt, count_alt, alt_dict, alt_rev_dict = build_dataset(
        altered_tokenized, vocab_size)
    # Train / valid / test split
    total_sents = len(orig)
    orig = np.array(orig)
    altered = np.array(altered)
    perm = np.random.permutation(total_sents)
    train_limit = int((0.97)*total_sents)
    valid_limit = int((0.98)*total_sents)
    orig_train = orig[perm[:train_limit]]
    orig_valid = orig[perm[train_limit:valid_limit]]
    orig_test = orig[perm[valid_limit:]]
    altered_train = altered[perm[:train_limit]]
    altered_valid = altered[perm[train_limit:valid_limit]]
    altered_test = altered[perm[valid_limit:]]

    # Create datafiles and vocab file
    orig_count = create_datafile(orig_train, "train_de_en.de")
    altered_count = create_datafile(altered_train, "train_de_en.en")
    orig_v_count = create_datafile(orig_valid, "valid_de_en.de")
    altered_v_count = create_datafile(altered_valid, "valid_de_en.en")
    orig_test_count = create_datafile(orig_test, "test_de_en.de")
    altered_test_count = create_datafile(altered_test, "test_de_en.en")
    
    
    vocab_count_orig = create_vocab_file(
        orig_rev_dict, "vocab_de_en_" + str(vocab_size) + ".de")
    vocab_count_alt = create_vocab_file(
        alt_rev_dict, "vocab_de_en_" + str(vocab_size) + ".en")
    
    print("src count: %d %d %d tgt count: %d %d %d, vocab_count: %d %d" % (
        orig_count, orig_v_count, orig_test_count,
        altered_count, altered_v_count, altered_test_count,
        vocab_count_orig, vocab_count_alt))

process("training-parallel-nc-v11", "de")
