import numpy as np

    # creates wordPairs for the Training & Validation Set
def createWordPairs(indexes, tokens, window_size):
    idx_pairs = []
        
    # for each sentence        
    for sentence in tokens.tokenzied_corpus:
#         if it can be indexed else skip to next iteration
        try:
            indices = [indexes.word2idx[word] for word in sentence]
        except Exception as e: print(e)
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))
    return np.array(idx_pairs)

    


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.train_loss = 0
        self.valid_loss = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count