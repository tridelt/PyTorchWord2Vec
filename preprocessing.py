import numpy as np

class Preprocess():

    def __init__(self, data_filename, threshold):
        self.data_dir = data_filename
        self.threshold = threshold
        self.buildingWordVariables()
    
    def tokenize_corpus(self, corpus):
        tokens = [x.split() for x in corpus]
        return tokens

    def onlyEmojiSequences(self, tokens):
        threshold_emojis = [x for x in tokens if len(x) > self.threshold]
        return threshold_emojis

    def buildingWordVariables(self):
        corpus = open(self.data_dir).read().splitlines()

        tokenized_corpus = self.tokenize_corpus(corpus)
#         should I set a threshold?
        emojiSequences = self.onlyEmojiSequences(tokenized_corpus)

        vocabulary = []
        for sentence in emojiSequences:
            for token in sentence:
                if token not in vocabulary:
                    vocabulary.append(token)

        self.tokenzied_corpus = tokenized_corpus
        self.word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
        self.idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
        self.vocabulary_size = len(vocabulary)