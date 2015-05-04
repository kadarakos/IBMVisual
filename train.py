from ibm_model import M1
import cPickle
from bravelearner import utils
from bravelearner.utils import BasicDataProvider, load_sets
import os
import scipy.sparse
import numpy as np
from collections import defaultdict
from operator import itemgetter
import sys
import argparse



def learn_embeddings(mode, dataset, split, outname,
                    online=True, weighted=False):
    """
    This function drives the word-embedding learnin experiment

    :param mode: string, 'monoling', 'img_id' or other
    Given 'monoling' it learns a monolingual translation table
    Given 'img_id' it learns a  distribution over imgids per word
    Given anything else it learns a distribution over image features per word

    :param dataset: string, Name of the dataset directory
    :param split: string, 'train', 'valid', 'test'
    :param outname: string, save the embeddings to file with outname 
    :return: None
    """

    print "Mode: ", mode
    print "Data set: ", dataset
    print
    print "Reading data"
    print

    # Translate from English from English
    if mode == 'monoling':
        datareader = BasicDataProvider(dataset)
        e = [x['tokens'] for x in datareader.iterSentences(split)]
        f = [x['tokens']+['NULL'] for x in datareader.iterSentences(split)] # adding NULL symbol

    # Translate from English to image features
    elif mode == 'vector':
        print "YEEE"
        e, f = utils.load_sets(split, dataset)
        f = np.asarray(f)
    else:
        print 'Unrecognized mode:', mode
        sys.exit()

    model = M1(f, e, online=online, weighted=weighted)

    # Running EM on the data and creating translation table
    print "Number of lines to process", len(e), len(f), '\n'
    model.iterate()
    translation_table = model.translation_table()

    # Convert the translation_table in the form of {word1: {0: 0.2, 1: 0.05 ..}, word2: ...}
    # to {word1: [0.2, 0.05, ...], word2: [...]}, write out both the word vectors and the translation table
    wordlist = translation_table.keys()
    num_words = str(len(wordlist))
    print "Creating word vectors for " + num_words + " words"
    word_vectors = {}
    counter = 0

    # Word vectors from translation table of image ids.
    if mode == 'monoling':
        for word in wordlist:
            counter += 1
            print 'At word', counter, "\r",
            word_vectors[word] = [translation_table[word].get(i, 0.0) for i in wordlist]
    # Word vectors from translation table of image features        
    else:
        for i in translation_table:
            counter += 1
            print 'At word', counter, "\r",
            word_vectors[i] = [translation_table[i].get(x, 0.0) for x in range(4096)]
        

    # Saving word-vectors to the given file
    #save_path = os.path.dirname(os.path.abspath("bravelearner"))
    #save_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    print "Writing word vectors to", outname
    cPickle.dump(word_vectors, open(outname, 'w'), protocol=2)
    #cPickle.dump(translation_table, open('translation_table.vectors', 'w'))
    print "Done"


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str , help='Either "monoling" or "vector"')
    parser.add_argument('mode', type=str , help='Either "batch" or "online"')
    parser.add_argument('dataset', type=str,  help='Path to dataset json')
    parser.add_argument('split', type=str,  help='"Train", "val" or "test"')
    parser.add_argument('o',help='Output filename to store the word embeddings')
    args = parser.parse_args()
    print args
    if args.model == "vector":
        weighted = True
    else:
        weighted = False

    learn_embeddings(args.model, args.dataset,
                    args.split, args.o, 
                    args.mode, weighted)