import glob
import random
import struct


class Vocab(object):

    def __init__(self, vocab_file, max_size):
        self.char_to_id = {}
        self.id_to_char = {}
        self.word_to_id = {}
        self._id_to_word = {}
        self._count = 0
