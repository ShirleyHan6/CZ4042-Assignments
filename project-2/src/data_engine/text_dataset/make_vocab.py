import os
import argparse
import pandas
import collections

parser = argparse.ArgumentParser()
parser.add_argument("-file_dir", default='ext', type=str, choices=['ext', 'abs'])
parser.add_argument("-alphabet", default=True)
parser.add_argument("-vocab_max_size", default=30000)
args = parser.parse_args()




class MakeVocab:
    def __init__(self, args):
        self.file_dir = os.path.join(args.file_dir, "train.csv")
        self.args = args

    def make_vocab(self):
        df = pandas.read_csv(args.file_path)
        vocab_counter = collections.Counter()

        if self.args.alphabet:
            for i, row in df.iterrows():
                text = row[2]
                alphabet_list = [i for i in text]
                tokens = [t.strip() for t in alphabet_list]
                tokens = [t for t in tokens if t != ""]
                vocab_counter.update(tokens)
        else:
            for i, row in df.iterrows():
                text = row[2]
                word_list = [i for i in text.split(" ")]
                tokens = [t.strip() for t in word_list]
                tokens = [t for t in tokens if t != ""]
                vocab_counter.update(tokens)

        file_prefix = "alphabet" if self.args.alphabet else "word"

        print("Writing {} vocab file...".format(file_prefix))

        with open(os.path.join(self.file_dir, "vocab_{}".format(file_prefix)), 'w') as writer:
            for word, count in vocab_counter:
                writer.write(word + ' ' + str(count) + '\n')

        print("Finished writing {} vocab file".format(file_prefix))








