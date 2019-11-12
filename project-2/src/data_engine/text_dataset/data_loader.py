from src.configs import configs
import src.data_engine.text_dataset.dataset as dataset
import numpy as np
import queue

class Example(object):
    def __init__(self, article, vocab):
        article_words = article.split()

        if len(article_words) > configs.MAX_ENC_STEP:
            article_words = article_words[:configs.MAX_ENC_STEP]

        self.enc_len = len(article_words)

        self.enc_iput = [vocab.word2id(w) for w in article_words]

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)


class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(dataset.PAD_TOKEN)
        self.init_encoder_seq(example_list)

    def init_encoder_seq(self, example_list):
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)

        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

# queue batch
class Batcher(object):
    BATCH_QUEUE_MAX=100

    def __init(self, data_path, vocab, mode, batch_size, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        if single_pass:
            self._num_example_q_threads = 1
            self.num_batch_q_threads = 1
            self.num_batch_q_threads = 1
            self._bucketing_cache_size = 1
