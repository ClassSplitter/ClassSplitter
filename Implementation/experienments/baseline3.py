from cluster import Member, Group, GroupInterface
from god_class import GodClass
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora, models, similarities
import re
import random
from lda import LDAHandlerSimple
from sklearn.metrics.pairwise import cosine_similarity
from timer import Timer


class Baseline3(GroupInterface):
    # 2014

    def __init__(self, god_class: GodClass):
        super().__init__(god_class)
        self.timer = Timer()
        self.SSMs = self.get_SSMs()
        self.CDMs = self.get_CDMs()
        self.CSMs = self.get_CSMs()
        self.couplings = self.get_couplings()
        self.graph, self.edges, self.weight = self.generate_graph()
        self.chains = self.graph.copy()

    def get_method_index(self, member: Member):
        if member in self.methods:
            return self.methods.index(member)
        else:
            raise IndexError

    def get_SSMs(self):
        ssms = [[0.0 for j in range(len(self.methods))] for i in range(len(self.methods))]
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                set1 = set(self.methods[i].get_visits())
                set2 = set(self.methods[j].get_visits())
                union = set1.union(set2)
                intersection = set1.intersection(set2)
                if len(union) > 0:
                    ssms[i][j] = len(intersection) / len(union)
                    ssms[j][i] = len(intersection) / len(union)
        return ssms

    def get_CDMs(self):
        cdms = [[0.0 for j in range(len(self.methods))] for i in range(len(self.methods))]
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                if len(self.methods[j].get_inner_invocations()) + len(self.methods[j].get_external_invocations()) > 0:
                    cdm1 = self.methods[j].get_inner_invocations().count(self.methods[i].get_name()) / (len(self.methods[j].get_inner_invocations()) + len(self.methods[j].get_external_invocations()))
                else:
                    cdm1 = 0.0
                if len(self.methods[i].get_inner_invocations()) + len(self.methods[i].get_external_invocations()) > 0:
                    cdm2 = self.methods[i].get_inner_invocations().count(self.methods[j].get_name()) / (
                                len(self.methods[i].get_inner_invocations()) + len(
                            self.methods[i].get_external_invocations()))
                else:
                    cdm2 = 0.0
                cdms[i][j] = max(cdm1, cdm2)
                cdms[j][i] = max(cdm1, cdm2)
        return cdms

    def split_camel_case(self, string):
        pattern = r'(?<!^)(?=[A-Z])'
        words = re.sub(pattern, ' ', string).split()
        return words

    def get_CSMs(self):
        documents = []
        for i in range(len(self.methods)):
            documents.append(re.sub(r"\W+", " ", self.methods[i].get_full_text()))
        o_texts = [[word for word in document.split()] for document in documents]
        texts = []
        for document in o_texts:
            new_document = []
            for word in document:
                new_document += self.split_camel_case(word)
            texts.append([w.lower() for w in new_document])
        lda = LDAHandlerSimple(texts)
        lda.train()
        distributions = lda.get_theta()
        similarity = cosine_similarity(distributions)
        return similarity.tolist()

        # dictionary = corpora.Dictionary(texts)
        # corpus = [dictionary.doc2bow(text) for text in texts]
        # lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=5)
        # corpus_lsi = lsi[corpus]
        # index = similarities.MatrixSimilarity(corpus_lsi)
        # sims = index[corpus_lsi].tolist()
        # for line in sims:
        #     for i in range(len(line)):
        #         line[i] = max(0.0, line[i])
        # return sims

    def get_couplings(self):
        w1 = 0.2
        w2 = 0.3
        w3 = 0.5
        couplings = [[0.0 for j in range(len(self.methods))] for i in range(len(self.methods))]
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                couplings[i][j] = w1 * self.SSMs[i][j] + w2 * self.CDMs[i][j] + w3 * self.CSMs[i][j]
                couplings[j][i] = w1 * self.SSMs[i][j] + w2 * self.CDMs[i][j] + w3 * self.CSMs[i][j]
                # if w1 * self.SSMs[i][j] + w2 * self.CDMs[i][j] + w3 * self.CSMs[i][j] >= 0.8:
                #     print("0.8!")
        return couplings

    def get_coupling(self, mem1, mem2):
        if isinstance(mem1, Member):
            index1 = self.get_method_index(mem1)
            index2 = self.get_method_index(mem2)
        else:
            index1 = mem1
            index2 = mem2
        return self.couplings[index1][index2]

    def generate_graph(self):
        graph = nx.Graph()
        nodes = self.methods[:]
        edges = []
        weights = []
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                edges.append((self.methods[i], self.methods[j]))
                weights.append(self.get_coupling(i, j))
        for i in range(len(nodes)):
            graph.add_node(nodes[i])
        for i in range(len(edges)):
            graph.add_edge(edges[i][0], edges[i][1], weight=weights[i])
        return graph, edges, weights

    def graph_to_chain(self):
        # 固定阈值
        threshold = 1.5 * sum(self.weight) / len(self.weight)
        print("average weight = %s" % str(sum(self.weight) / len(self.weight)))
        self.chains = self.graph.copy()
        for i in range(len(self.edges)):
            if threshold > self.chains.get_edge_data(self.edges[i][0], self.edges[i][1])['weight']:
                self.chains.remove_edge(self.edges[i][0], self.edges[i][1])

    def get_chain_number(self):
        chain_number = 0
        for component in nx.connected_components(self.chains):
            if len(component) >= 3:
                chain_number += 1
        return chain_number

    def graph_to_2chain(self):
        # 二分查找满足得到2个链的阈值
        test_thresholds = [i * 0.01 for i in range(0, 100)]
        thresholds = [0.0, 0.9, 0.45]
        for threshold in test_thresholds:
            self.chains = self.graph.copy()
            for i in range(len(self.edges)):
                if threshold > self.chains.get_edge_data(self.edges[i][0], self.edges[i][1])['weight']:
                    self.chains.remove_edge(self.edges[i][0], self.edges[i][1])
            if self.get_chain_number() == 2:
                return
            elif self.get_chain_number() < 2 and threshold > thresholds[0]:
                thresholds[0] = threshold
            elif self.get_chain_number() > 2 and threshold < thresholds[1]:
                thresholds[1] = threshold
        # thresholds[2] = (thresholds[0] + thresholds[1]) / 2
        # while True:
        #     self.chains = self.graph.copy()
        #     for i in range(len(self.edges)):
        #         if thresholds[2] > self.chains.get_edge_data(self.edges[i][0], self.edges[i][1])['weight']:
        #             self.chains.remove_edge(self.edges[i][0], self.edges[i][1])
        #     if self.get_chain_number() == 2:
        #         break
        #     elif self.get_chain_number() > 2:
        #         thresholds[1] = thresholds[2]
        #         thresholds[2] = (thresholds[0] + thresholds[1]) / 2
        #     else:
        #         thresholds[0] = thresholds[2]
        #         thresholds[2] = (thresholds[0] + thresholds[1]) / 2
        self.graph_to_chain()

    def original_graph_to_chain(self, t=0.4):
        # get quartiles q1, q2, q3
        weights = self.weight[:]
        weights.sort()
        threshold = t
        self.chains = self.graph.copy()
        for i in range(len(self.edges)):
            if threshold > self.chains.get_edge_data(self.edges[i][0], self.edges[i][1])['weight']:
                self.chains.remove_edge(self.edges[i][0], self.edges[i][1])

    def compute_group_coupling(self, group1: Group, group2: Group):
        m1 = len(group1)
        m2 = len(group2)
        couplings = []
        for i in range(m1):
            method1 = group1.members[i]
            for j in range(m2):
                method2 = group2.members[j]
                coupling = self.get_coupling(method1, method2)
                couplings.append(coupling)
        return sum(couplings) / (m1 * m2)

    def get_smallest_group_size(self):
        size = 1000
        for group in self.groups:
            size = min(size, len(group))
        return size

    def chain_to_groups(self):
        for component in nx.connected_components(self.chains):
            new_group = Group()
            for member in component:
                new_group.append(member)
            self.groups.append(new_group)
        while len(self.groups) > 2:
            t = self.get_smallest_group_size()
            for i in range(len(self.groups)):
                if len(self.groups[i]) <= t:
                    small_group = self.groups.pop(i)
                    break
            move_group_indexs = []
            move_group_couplings = []
            for i in range(len(self.groups)):
                if len(self.groups[i]) >= t:
                    move_group_indexs.append(i)
                    move_group_couplings.append(self.compute_group_coupling(small_group, self.groups[i]))
            max_index = move_group_indexs[move_group_couplings.index(max(move_group_couplings))]
            self.groups[max_index] = self.groups[max_index] + small_group
            if self.get_chain_number() == 2:
                t = 3

    def original_chain_to_groups(self):
        for component in nx.connected_components(self.chains):
            new_group = Group()
            for member in component:
                new_group.append(member)
            self.groups.append(new_group)
        # min_length = 3
        # t = self.get_smallest_group_size()
        # while t <= min_length and len(self.groups) > 2:
        #     for i in range(len(self.groups)):
        #         if len(self.groups[i]) <= t:
        #             small_group = self.groups.pop(i)
        #             break
        #     move_group_indexes = []
        #     move_group_couplings = []
        #     for i in range(len(self.groups)):
        #         if len(self.groups[i]) >= t:
        #             move_group_indexes.append(i)
        #             move_group_couplings.append(self.compute_group_coupling(small_group, self.groups[i]))
        #     max_index = move_group_indexes[move_group_couplings.index(max(move_group_couplings))]
        #     self.groups[max_index] = self.groups[max_index] + small_group
        #     self.remove_empty_group()
        #     t = self.get_smallest_group_size()

    def draw_graph(self):
        pos = nx.spring_layout(self.chains)
        nx.draw(self.chains, pos, with_labels=True)
        # edge_labels = nx.get_edge_attributes(self.chains, "weight")
        # nx.draw_networkx_edge_labels(self.chains, pos, edge_labels=edge_labels)
        print(self.SSMs)
        print(self.CDMs)
        print(self.CSMs)

    def handel_fields(self, use_position=False):
        fields = self.get_fields()
        single_fields = []
        for field in fields:
            visit_times = []
            for i in range(len(self.groups)):
                visit_time = 0
                for method in self.groups[i]:
                    visit_time += 1 if method.get_visit_times(field.get_name()) > 0 else 0
                visit_times.append(visit_time)
            if sum(visit_times) > 0:
                self.groups[visit_times.index(max(visit_times))].append(field)
            else:
                single_fields.append(field)
        for field in single_fields:
            if use_position:
                distance_scores = []
                for i in range(len(self.groups)):
                    score = 0
                    for member in self.groups[i]:
                        if member.is_field():
                            score += 1.0 / abs(member.get_index() - field.get_index()) if abs(
                                member.get_index() - field.get_index()) < 10 else 0.0
                        if member.is_method():
                            if member.get_index() > field.get_index():
                                score += 1.0 / abs(member.get_index() - field.get_index()) if abs(
                                    member.get_index() - field.get_index()) < 10 else 0.0
                    distance_scores.append(score)
                self.groups[distance_scores.index(max(distance_scores))].append(field)
            else:
                self.groups[random.randint(0, len(self.groups) - 1)].append(field)