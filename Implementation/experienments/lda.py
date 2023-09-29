import numpy as np
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from god_class import GodClass
import re
from gensim import corpora, models, similarities


class LDAHandler:

    def __init__(self, god_class: GodClass, use_gpt_texts=False):
        self.god_class = god_class
        # self.common_texts = self.get_common_texts(use_gpt_texts)
        self.common_texts = self.get_full_texts(use_gpt_texts)
        self.dictionary = Dictionary(self.common_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.common_texts]
        self.lda = None
        self.k = None

    def train(self, k=10):
        self.k = k
        self.lda = LdaModel(self.corpus, num_topics=k, alpha='auto', eta='auto', iterations=500, passes=20)

    def get_common_texts(self, use_gpt_texts):
        texts = []
        for i in range(len(self.god_class.data_frame)):
            if use_gpt_texts:
                try:
                    text_str = str(self.god_class.data_frame.at[i, 'document']) + \
                               str(self.god_class.data_frame.at[i, 'gpt text'])
                except KeyError:
                    text_str = str(self.god_class.data_frame.at[i, 'document'])
                    print(self.god_class.data_path + " : no gpt texts found")
            else:
                text_str = str(self.god_class.data_frame.at[i, 'document'])
            text_str = remove_stopwords(text_str)
            text = simple_preprocess(text_str)
            texts.append(text)
        return texts

    def get_full_texts(self, use_gpt_texts):
        documents = [re.sub(r"\W+", " ", document) for document in self.god_class.data_frame['full text'].to_list()]
        if use_gpt_texts:
            for i in range(len(self.god_class)):
                try:
                    gpt_text = re.sub(r"\W+", " ", str(self.god_class.data_frame.at[i, 'gpt text']))
                    documents[i] += " " + gpt_text
                except KeyError:
                    print(self.god_class.data_path + " : no gpt texts found")
        texts = [simple_preprocess(remove_stopwords(document)) for document in documents]
        return texts

    def get_theta(self):
        theta = []
        topics = self.lda.get_topics()
        for i in range(len(self.god_class.data_frame)):
            topic_distribution = [0.0 for i in range(self.k)]
            vector = self.lda[self.corpus[i]]
            for t in vector:
                topic_distribution[t[0]] = t[1]
            theta.append(topic_distribution)
        return theta


class LSIHandler:

    def __init__(self, god_class: GodClass, use_gpt_texts=False):
        self.god_class = god_class
        self.common_texts = self.get_common_texts(use_gpt_texts)
        self.dictionary = corpora.Dictionary(self.common_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.common_texts]
        self.lsi = None
        self.k = None

    def get_common_texts(self, use_gpt_texts):
        texts = []
        for i in range(len(self.god_class.data_frame)):
            if use_gpt_texts:
                try:
                    text_str = str(self.god_class.data_frame.at[i, 'document']) + \
                               str(self.god_class.data_frame.at[i, 'gpt text'])
                except KeyError:
                    text_str = str(self.god_class.data_frame.at[i, 'document'])
                    print(self.god_class.data_path + " : no gpt texts found")
            else:
                text_str = str(self.god_class.data_frame.at[i, 'document'])
            text_str = remove_stopwords(text_str)
            text = simple_preprocess(text_str)
            texts.append(text)
        return texts

    def train(self, k=10):
        self.lsi = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=k)
        self.k = k

    def get_theta(self):
        corpus_lsi = self.lsi[self.corpus]
        index = similarities.MatrixSimilarity(corpus_lsi)
        return index.index.tolist()


class LDAHandlerSimple:

    def __init__(self, texts):
        # self.common_texts = self.get_common_texts(use_gpt_texts)
        self.common_texts = texts
        self.dictionary = Dictionary(self.common_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.common_texts]
        self.lda = None
        self.k = None

    def train(self, k=10):
        self.k = k
        self.lda = LdaModel(self.corpus, num_topics=k, alpha='auto', eta='auto', iterations=200, passes=10)

    def get_theta(self):
        theta = []
        topics = self.lda.get_topics()
        for i in range(len(self.common_texts)):
            topic_distribution = [0.0 for i in range(self.k)]
            vector = self.lda[self.corpus[i]]
            for t in vector:
                topic_distribution[t[0]] = t[1]
            theta.append(topic_distribution)
        return theta
