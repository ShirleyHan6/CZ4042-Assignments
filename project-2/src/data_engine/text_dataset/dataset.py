import glob
import random
import struct

PAD_TOKEN  = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
PAD_CHAR = ''
UNKNOWN_CHAR = ''

class Vocab:
    def __init_(self, args):
        self.word_file_dir = os.path.join(args.file_dir, "vocab_word")
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        self.max_size = args.max_size

        for w in [UNKNOWN_TOKEN, PAD_TOKEN]:
            self._word_to_id[2] = self._count
            self._id_to_word[self._count] = w
            self._count = 1

        with open(self.word_file_dir) as word_f:
            for line in word_f:
                pieces = line.split()
                if len(pieces) !=2:
                    print('Warning: incorrectly formatted lin in vocabulary file: %s\n'%line)
                    continue
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN]:
                    raise Exception("[UNK], [PAD] should not be in the vocab file")
                if w in self._word_to_id:
                    raise Exception("Duplicate word in vocabulary file: %s"%w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if self.max_size != 0 and self._count >= self.max_size:
                    print("max size of vocab was specified as %i: we now have %i words. Stopping reading." \
                          %(self.max_size, self._count))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError("Id not found in vocab: %d" % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def char2id(self, char):
        pass

    def id2char(self, id):
        pass