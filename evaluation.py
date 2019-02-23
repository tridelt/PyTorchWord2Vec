from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from numpy  import array
import re
import codecs
import numpy as np
from scipy import stats

class Metrics():
    
    def __init__(self, trainedWeights, preprocessValues):
#        instantiating the instance variables
        self.trainedWeights = trainedWeights
        self.word2idx = preprocessValues.word2idx
        self.corpus = self.retrieveCorpus()
        self.annotator_similarity_score_508 = self.createAnnotators()
        self.google_sense_labels_score_508 = self.createGoogle_Sense()
        self.glyph_pairs_1016 = self.createGlyph_pairs_1016()
        self.goldstandard = []
        self.selftrained = []
        self.google_sense_labels = []
#        computing the metrics
        self.getSimilarities()
        self.computeMetrics()
    
#    loads the corpus for all the following methods
    def retrieveCorpus(self):
        corpus_filename = './data/EmoSim508.json'
        return open(corpus_filename).read()
    
#    finds the relevant values from the corpus
    def createAnnotators(self):
        return list(array(re.findall('(?<=_Annotator_Agreement": )(.*?)(?=\})', self.corpus)))
    
#    extracts relevant data from the loaded corpus
    def createGoogle_Sense(self):
        return list(array(re.findall('(?<=Google_Sense_Label": )(.*?)(?=\,)', self.corpus)))
    def createGlyph_pairs_1016(self):
        unicode_pairs_1016 = re.findall('(?<=unicodelong": "\\\)(.*?)(?=")', self.corpus)
        return [codecs.decode(unicode_pairs_1016[x].replace(str('\\\\'),str('\\')).replace('_',''), 'unicode_escape') for x in range(len(unicode_pairs_1016))]

#    saves the SimilarityScore only if cosineSimilarity was successfully computed for that emoji-pair
    def getSimilarities(self):
        for x in range(len(self.annotator_similarity_score_508)):
            cosineSimilarity = None
            emoji1 = self.glyph_pairs_1016.pop(0)
            emoji2 = self.glyph_pairs_1016.pop(0)
            
            try:
                cosineSimilarity = cosine_similarity(self.trainedWeights[self.word2idx[emoji1]].reshape(-1, self.trainedWeights.shape[1]), self.trainedWeights[self.word2idx[emoji2]].reshape(-1, self.trainedWeights.shape[1]))[0][0]
            except:
                print('the cosine similarity between ' + emoji1 + ' and ' + emoji2 + ' could not be computed.')

            if(cosineSimilarity is not None):
                self.goldstandard.append(self.annotator_similarity_score_508.pop(0))
                self.selftrained.append(cosineSimilarity)
                self.google_sense_labels.append(float(self.google_sense_labels_score_508.pop(0)))
            else:
                self.annotator_similarity_score_508.pop(0)
                float(self.google_sense_labels_score_508.pop(0))

#    computes the actual metrics
    def computeMetrics(self):
        # skalierter GoldStandard
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_goldstandard = min_max_scaler.fit_transform(np.asarray(self.goldstandard).reshape(-1, 1))
        print()

        # computation of SPEARRANK CORRELATION COEFFICIENT
        meinSPEARMAN = stats.spearmanr(self.goldstandard, self.selftrained)
        seinSPEARMAN = stats.spearmanr(self.goldstandard, self.google_sense_labels)
        print('mein Spearman: {}'.format(meinSPEARMAN.correlation))
        print('sein Spearman: {}'.format(seinSPEARMAN.correlation))

        # computation of MAE
        meinMAE = mean_absolute_error(scaled_goldstandard, min_max_scaler.fit_transform(np.asarray(self.selftrained).reshape(-1, 1)))
        seinMAE = mean_absolute_error(scaled_goldstandard, self.google_sense_labels)
        print('mein MAE ist {}'.format(meinMAE))
        print('sein MAE ist {}'.format(seinMAE))

        # computation of MSE
        meinMSE = mean_squared_error(scaled_goldstandard, min_max_scaler.fit_transform(np.asarray(self.selftrained).reshape(-1, 1)))
        seinMSE = mean_squared_error(scaled_goldstandard, self.google_sense_labels)
        print('mein MSE ist {}'.format(meinMSE))
        print('sein MSE ist {}'.format(seinMSE))




