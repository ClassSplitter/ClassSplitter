import pickle

import pandas.core.frame
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics.pairwise
import copy

from god_class import GodClass
from lda import LDAHandler, LSIHandler
from sklearn.cluster import KMeans
from gensim import corpora, models, similarities
import re
import networkx as nx
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import os
from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from timer import Timer


# os.environ['TRANSFORMERS_CACHE'] = 'F:\\dependence\\transformers_cache'
st5model = SentenceTransformer("krlvi/sentence-t5-base-nlpl-code_search_net")
bgemodel = SentenceTransformer('BAAI/bge-base-en')
# word2vec_model = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt')


class Member:

    def __init__(self, data: pandas.core.frame.DataFrame, distribution=0):
        self.data = data
        self.distribution = distribution

    # def __eq__(self, other):
    #     if isinstance(other, Member):
    #         if other.get_index() == self.get_index() and other.get_name() == self.get_name():
    #             return True
    #     return False

    def set_distribution(self, distribution):
        self.distribution = distribution

    def get_index(self):
        return int(self.data.at[self.data.iloc[0].name, 'index'])

    def get_name(self):
        return str(self.data.at[self.data.iloc[0].name, 'name'])

    def is_method(self):
        if self.data.at[self.data.iloc[0].name, 'type'] == 'Method':
            return True
        else:
            return False

    def is_field(self):
        if self.data.at[self.data.iloc[0].name, 'type'] == 'Field':
            return True
        else:
            return False

    def is_moved(self):
        if self.data.at[self.data.iloc[0].name, 'moved']:
            return True
        else:
            return False

    def is_removed(self):
        if self.data.at[self.data.iloc[0].name, 'removed']:
            return True
        else:
            return False

    def get_document(self):
        return str(self.data.at[self.data.iloc[0].name, 'document'])

    def get_distribution(self):
        return self.distribution

    def get_gpt_text(self):
        try:
            text = self.data.at[self.data.iloc[0].name, 'gpt text']
        except KeyError:
            tqdm.write("no gpt text.")
            return None
        if text == '':
            tqdm.write("empty gpt text.")
            return None
        return str(self.data.at[self.data.iloc[0].name, 'gpt text'])

    def get_code_summary(self):
        try:
            text = self.data.at[self.data.iloc[0].name, 'code summary']
        except KeyError:
            tqdm.write("no code summary.")
            return None
        if text == '':
            tqdm.write("empty code summary.")
            return None
        return str(self.data.at[self.data.iloc[0].name, 'code summary'])

    def get_inner_invocations(self):
        if str(self.data.at[self.data.iloc[0].name, 'inner invocations']) == '':
            return []
        else:
            return str(self.data.at[self.data.iloc[0].name, 'inner invocations']).split()

    def get_external_invocations(self):
        if str(self.data.at[self.data.iloc[0].name, 'external invocations']) == '':
            return []
        else:
            return str(self.data.at[self.data.iloc[0].name, 'external invocations']).split()

    def get_visits(self):
        if str(self.data.at[self.data.iloc[0].name, 'visits']) == 'nan':
            return []
        else:
            return str(self.data.at[self.data.iloc[0].name, 'visits']).split()

    def get_full_text(self):
        return str(self.data.at[self.data.iloc[0].name, 'full text'])

    def is_gpt_response_extract(self):
        try:
            if self.data.at[self.data.iloc[0].name, 'gpt response']:
                return True
            else:
                return False
        except KeyError:
            return False

    def get_visit_times(self, name: str):
        if str(self.data.at[self.data.iloc[0].name, 'visits']) == 'nan':
            return 0
        visit_full_names = str(self.data.at[self.data.iloc[0].name, 'visits']).split()
        visits = [string.split('+')[1] for string in visit_full_names]
        return visits.count(name)

    def compute_structure(self, group):
        member_inner_invocations = self.get_inner_invocations()
        group_names = group.get_names()
        inner_invocation_by_group = 0
        if len(member_inner_invocations) > 0:
            for name in member_inner_invocations:
                if name in group_names:
                    inner_invocation_by_group += 1

        member_external_invocations = [c.split('+')[0] for c in self.get_external_invocations()]
        group_external_invocations = []
        for mem in group.members:
            group_external_invocations += [c.split('+')[0] for c in mem.get_external_invocations()]
        external_invocation_by_group = 0
        for origin_class in member_external_invocations:
            external_invocation_by_group += group_external_invocations.count(origin_class)

        return inner_invocation_by_group, external_invocation_by_group


class Group:

    def __init__(self):
        self.members: list[Member] = []
        self.indexes: list[int] = []
        self.k = None

    def __add__(self, other):
        self.members += other.members
        self.indexes += other.indexes
        self.indexes.sort()
        # if self.k is None and len(self.members) > 0 and self.members[0].distribution != 0:
        #     self.k = len(self.members[0].distribution)
        return self

    def __contains__(self, item):
        if isinstance(item, Member):
            if item in self.members:
                return True
        elif isinstance(item, int):
            if item in self.indexes:
                return True
        return False

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        return self.members[index]

    def remove(self, item):
        if item not in self:
            return False
        try:
            if isinstance(item, Member):
                self.members.remove(item)
                this_index = item.get_index()
                self.indexes.remove(this_index)
            elif isinstance(item, int):
                self.indexes.remove(item)
                for mem in self.members:
                    if mem.get_index() == item:
                        this_member = mem
                self.members.remove(this_member)
            return True
        except Exception:
            return False

    def append(self, member: Member):
        self.members.append(member)
        self.indexes.append(member.get_index())
        self.indexes.sort()
        # if self.k is None and len(self.members) > 0 and self.members[0].distribution != 0:
        #     self.k = len(self.members[0].distribution)

    def get_names(self):
        names = []
        for member in self.members:
            names.append(member.get_name())
        return names

    def get_average_index(self):
        index_sum = 0.0
        for mem in self.members:
            index_sum += mem.get_index()
        return index_sum / len(self.members)

    def get_moved_number(self):
        moved_number = 0
        for mem in self.members:
            if mem.is_moved():
                moved_number += 1
        return moved_number

    def get_removed_number(self):
        removed_number = 0
        for mem in self.members:
            if mem.is_removed():
                removed_number += 1
        return removed_number

    def get_moved_percentage(self):
        return self.get_moved_number() / len(self.members)

    def get_average_distribution(self, member: Member = None):
        if len(self.members) == 0:
            return [0.0 for i in range(self.k)]
        distribution_sum = [0.0 for i in range(self.k)]
        exception_flag = 0
        for mem in self.members:
            if member is None or member.get_name() != mem.get_name():
                for i in range(len(distribution_sum)):
                    distribution_sum[i] += mem.distribution[i]
            else:
                exception_flag = 1
        dist_len = len(self.members) - exception_flag
        if dist_len == 0:
            return [0.0 for i in range(self.k)]
        else:
            return [i / dist_len for i in distribution_sum]

    def compute_structure(self, member: Member):
        member_inner_invocations = member.get_inner_invocations()
        group_names = self.get_names()
        inner_invocation_by_group = 0
        if len(member_inner_invocations) > 0:
            for name in member_inner_invocations:
                if name in group_names:
                    inner_invocation_by_group += 1

        member_external_invocations = [c.split('+')[0] for c in member.get_external_invocations()]
        group_external_invocations = []
        for mem in self.members:
            if mem.get_name() != member.get_name():
                group_external_invocations += [c.split('+')[0] for c in mem.get_external_invocations()]
        external_invocation_by_group = 0
        for origin_class in member_external_invocations:
            external_invocation_by_group += group_external_invocations.count(origin_class)

        member_field_visits = set(member.get_visits())
        group_field_relations = []
        for mem in self.members:
            if mem.get_name() != member.get_name():
                mem_field_visits = set(mem.get_visits())
                union = mem_field_visits.union(member_field_visits)
                intersection = mem_field_visits.intersection(member_field_visits)
                if len(union) > 0:
                    group_field_relations.append(len(intersection) / len(union))
                else:
                    group_field_relations.append(0.0)
        if len(group_field_relations) == 0:
            visit_coincide_by_group = 0
        else:
            visit_coincide_by_group = sum(group_field_relations) / len(group_field_relations)

        return inner_invocation_by_group, external_invocation_by_group, visit_coincide_by_group

    def compute_distance(self, member: Member):
        distances = []
        for mem in self.members:
            distance = abs(mem.get_index() - member.get_index())
            if distance != 0:
                distances.append(distance)
        if len(distances) != 0:
            reciprocal_distance = sum([(1.0 / x) ** 2 for x in distances])
            average_distance = sum(distances) / len(distances)
            return reciprocal_distance, average_distance
        else:
            return 0, 1000

    def compute_invocation(self, member: Member):
        inner_invocation = self.compute_structure(member)[0]
        if len(member.get_inner_invocations()) == 0:
            return 0.0
        else:
            return inner_invocation / len(member.get_inner_invocations())

    def get_membership(self, member: Member):
        return self.compute_distance(member)[0] + self.compute_invocation(member)

    def get_score(self, member: Member):
        len_member = len(self.members)
        if self.__contains__(member):
            len_member -= 1
        location_score = self.compute_distance(member)[0] / 4.0
        if len_member <= 2:
            location_score = location_score / 2

        inner_invocation, external_invocation, visits_score = self.compute_structure(member)
        if len(member.get_inner_invocations()) > 0:
            structure_inner_score = inner_invocation / len(member.get_inner_invocations())
        else:
            structure_inner_score = 0.0
        if external_invocation > 0:
            structure_external_score = 1.0
        else:
            structure_external_score = 0.0

        semantics_score = sum([i * j for i, j in zip(self.get_average_distribution(member), member.distribution)])

        return 2.0 * location_score + 2.0 * structure_inner_score + structure_external_score + visits_score + 2.0 * semantics_score


class GroupInterface:
    def __init__(self, god_class: GodClass):
        self.groups: list[Group] = []
        self.figure_number = 0
        self.god_class = god_class
        self.members = self.get_members()  # List{Member}
        self.methods = self.get_methods()  # List{Member}

    def get_members(self):
        members = []
        for i in range(len(self.god_class.data_frame)):
            member = Member(pd.DataFrame(self.god_class.data_frame.iloc[i]).T)
            members.append(member)
        return members

    def get_methods(self):
        methods = []
        for mem in self.members:
            if mem.is_method():
                methods.append(mem)
        methods.sort(key=lambda x: x.get_index())
        return methods

    def get_fields(self):
        fields = []
        for mem in self.members:
            if mem.is_field():
                fields.append(mem)
        fields.sort(key=lambda x: x.get_index())
        return fields

    def remove_empty_group(self):
        for i in range(len(self.groups) - 1, -1, -1):
            if self.groups[i] is None or len(self.groups[i]) == 0:
                self.groups.pop(i)

    def sort_group_by_position(self):
        self.groups.sort(key=lambda x: x.get_average_index())

    def draw_groups(self, fixed=False, field=False, index=''):
        plt.figure('cluster group' + index + '(' + str(self.figure_number) + ")")
        self.figure_number += 1
        self.remove_empty_group()
        groups_to_show = self.groups[:]
        groups_to_show.sort(key=lambda x: x.get_average_index())
        drawed_mems = []
        for i in range(len(groups_to_show)):
            if fixed:
                value_x = 0
            else:
                value_x = i + 1
            for mem in groups_to_show[i].members:
                drawed_mems.append(mem)
                value_y = mem.get_index()
                if mem.is_field():
                    marker = "^"
                    if mem.is_moved():
                        text_color = (0.6, 0.0, 0.0)
                    else:
                        text_color = (0.0, 0.6, 0.0)
                elif mem.is_method():
                    marker = "o"
                    if mem.is_moved():
                        text_color = (0.9, 0.0, 0.0)
                    else:
                        text_color = (0.0, 0.9, 0.0)
                else:
                    marker = "s"
                    if mem.is_moved():
                        text_color = (1.0, 0.5, 0.5)
                    else:
                        text_color = (0.5, 0.5, 0.5)
                plt.scatter(value_x, value_y, color=text_color, s=60, marker=marker)
                plt.text(value_x + 0.02 * (len(self.groups) - 1), value_y, mem.get_name(), color=text_color,
                         horizontalalignment='left',
                         verticalalignment='center', fontsize=10)
        value_x = 0
        for mem in self.members:
            if mem in drawed_mems:
                continue
            value_y = mem.get_index()
            if mem.is_field():
                marker = "^"
                if mem.is_moved():
                    text_color = (0.6, 0.0, 0.0)
                else:
                    text_color = (0.0, 0.6, 0.0)
            elif mem.is_method():
                marker = "o"
                if mem.is_moved():
                    text_color = (0.9, 0.0, 0.0)
                else:
                    text_color = (0.0, 0.9, 0.0)
            else:
                marker = "s"
                if mem.is_moved():
                    text_color = (1.0, 0.5, 0.5)
                else:
                    text_color = (0.5, 0.5, 0.5)
            plt.scatter(value_x, value_y, color=text_color, s=60, marker=marker)
            plt.text(value_x + 0.02 * (len(self.groups) - 1), value_y, mem.get_name(), color=text_color,
                     horizontalalignment='left',
                     verticalalignment='center', fontsize=10)
        plt.gca().invert_yaxis()
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel("group tag")
        ax.set_ylabel("member declaration sequence")
        # ax.set_xticks(range(min(clustering), max(clustering) + 1))
        # ax.set_xticklabels(['#' + str(i) for i in range(min(clustering), max(clustering) + 1)])
        # ax.set_yticks(range(len(data_reader.names)))
        plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
        print("MoJoFM: %.3f" % (self.get_mojofm(field=field)))

    def compute_tp_tn_fp_fn(self):
        if len(self.groups) != 2:
            print('group number error: %d in %s' % (len(self.groups), str(self.god_class.get_index())))
            return 0.0, 0.0, 0.0, 0.0
        if self.groups[0].get_moved_percentage() > self.groups[1].get_moved_percentage():
            split_group = 0
        else:
            split_group = 1
        tp = self.groups[split_group].get_removed_number()
        fn = self.groups[1 - split_group].get_removed_number()
        fp = len(self.groups[split_group]) - tp
        tn = len(self.groups[1 - split_group]) - fn
        return tp, tn, fp, fn

    def get_precision_recall(self):
        tp, tn, fp, fn = self.compute_tp_tn_fp_fn()
        try:
            return tp / (tp + fp), tp / (tp + fn)
        except ZeroDivisionError:
            print('zero division error in %s' % str(self.god_class.get_index()))
            return 0.0, 0.0

    def get_accuracy(self):
        tp, tn, fp, fn = self.compute_tp_tn_fp_fn()
        try:
            return (tp + tn) / (tp + tn + fp + fn)
        except ZeroDivisionError:
            print('zero division error in %s' % str(self.god_class.get_index()))
            return 0.0

    # def get_mojofm(self):
    #     if len(self.groups) != 2:
    #         print('group number error in %s' % str(self.god_class.get_index()))
    #         return 0.0
    #     max_mno = math.floor(0.5 * len(self.methods))
    #     g0_move = self.groups[0].get_removed_number()
    #     g0_stay = len(self.groups[0]) - g0_move
    #     g1_move = self.groups[1].get_removed_number()
    #     g1_stay = len(self.groups[1]) - g1_move
    #     mno = min(g0_move + g1_stay, g0_stay + g1_move)
    #     mojofm = 1.0 - mno / max_mno
    #     return mojofm

    def get_mojofm(self, field=False):
        # if len(self.groups) < 2:
        #     print('group number error in %s, only 1 group' % str(self.god_class.get_index()))
        self.group_to_ref()
        self.example_to_ref(field=field)
        f = os.popen('java mojo.MoJo group.ref example.ref -fm')
        r_str = f.read()
        result = float(r_str.strip())
        f.close()
        return max(0.0, (result - 50.0) / 50.0)
        # return result / 100.0

    def group_to_ref(self):
        str_builder = ""
        for i in range(len(self.groups)):
            c_name = 'c' + str(i)
            for j in range(len(self.groups[i])):
                str_builder += 'contain ' + c_name + ' ' + str(self.groups[i][j].get_index()) + '\n'
        f = open('group.ref', 'w')
        f.write(str_builder)
        f.close()

    def example_to_ref(self, field):
        str_builder = ""
        for i in range(len(self.members)):
            c_name = 'c' + str(self.members[i].is_removed())
            if self.members[i].is_method() or (self.members[i].is_field() and field):
                str_builder += 'contain ' + c_name + ' ' + str(self.members[i].get_index()) + '\n'
        f = open('example.ref', 'w')
        f.write(str_builder)
        f.close()

    def get_field_mojofm(self):
        # if len(self.groups) < 2:
        #     print('group number error in %s, only 1 group' % str(self.god_class.get_index()))
        str_builder = ""
        for i in range(len(self.groups)):
            c_name = 'c' + str(i)
            for j in range(len(self.groups[i])):
                if self.groups[i][j].is_field():
                    str_builder += 'contain ' + c_name + ' ' + str(self.groups[i][j].get_index()) + '\n'
        if str_builder == "":
            # print("no field in class %s" % str(self.god_class.get_index()))
            return 0.5
        f = open('group.ref', 'w')
        f.write(str_builder)
        f.close()
        str_builder = ""
        for i in range(len(self.members)):
            c_name = 'c' + str(self.members[i].is_removed())
            if self.members[i].is_field():
                str_builder += 'contain ' + c_name + ' ' + str(self.members[i].get_index()) + '\n'
        if str_builder == "":
            # print("no field in class %s" % str(self.god_class.get_index()))
            return 0.5
        f = open('example.ref', 'w')
        f.write(str_builder)
        f.close()
        f = os.popen('java mojo.MoJo group.ref example.ref -fm')
        r_str = f.read()
        result = float(r_str.strip())
        f.close()
        return max(0.0, (result - 50.0) / 50.0)


def split_camel_case(string):
    pattern = r'(?<!^)(?=[A-Z])'
    words = re.sub(pattern, ' ', string).split()
    return words


cache = True
# cache = False


class Cluster(GroupInterface):

    def __init__(self, god_class: GodClass):
        super().__init__(god_class)
        data_path = 'cluster_pickle_data3/' + str(self.god_class.get_last_index())
        # data_path = 'cluster_pickle_data4/' + str(self.god_class.get_last_index())
        try:
            if not cache:
                raise OSError
            with open(data_path, 'rb') as f:
                history_data: ClusterDataContainer = pickle.load(f)
            self.distribution = history_data.distribution
            self.k = 0
            self.distances = history_data.distances
            self.SSMs = history_data.SSMs
            self.CDMs = history_data.CDMs
            self.CSMs = history_data.CSMs
            self.code_summary_similarity = history_data.code_summary_similarity
            # self.timer = history_data.timer
            self.timer = Timer()
        except OSError:
            self.timer = Timer()
            self.distances = self.get_distances()
            self.timer.record_past('distance')
            self.SSMs = self.get_SSMs()
            self.CDMs = self.get_CDMs()
            self.timer.record_past('metrics')
            self.distribution, self.k = self.get_distribution()
            self.timer.record_past('fullcode encode')
            self.CSMs = self.get_CSMs()
            self.timer.record_past('fullcode sim calculation')
            self.code_summary_similarity = self.get_summary_similarity()
            self.timer.record_past('summary sim calculation')
            with open(data_path, 'wb') as f:
                pickle.dump(
                    ClusterDataContainer(self.god_class, self.distribution, self.distances, self.SSMs, self.CDMs,
                                         self.CSMs, self.code_summary_similarity, self.timer), f)

        # self.lda_distribution, self.k = self.get_lda()
        # self.lda_relation = self.get_lda_relation()
        # self.externals = self.get_externals()
        # self.groups = self.k_means_get_groups()  # List{4.0Group}
        # self.groups = self.get_init_groups()
        # self.parameters_weight: tuple[float, float, float, float, float] = (4.6, 2.9, 3.8, 0.9, 0.0)
        self.graph = None
        self.edges = None
        self.weight = None
        # self.chains = None

    def graph_to_groups(self):
        self.graph, self.edges, self.weight = self.generate_graph()
        # self.chains = self.graph.copy()
        # self.graph_to_chains()
        self.chain_to_groups()

    def get_distribution(self):
        # embeddings = embedding_st5model(tuple([method.get_full_text() for method in self.methods]))
        # for i in range(len(self.methods)):
        #     self.methods[i].set_distribution(embeddings[i])
        embeddings = st5model.encode([method.get_full_text() for method in self.methods])
        return embeddings, 0

    def get_lda(self):
        lda_handler = LDAHandler(self.god_class, use_gpt_texts=True)
        k = max(2, math.ceil(self.god_class.get_method_number() / 6))
        lda_handler.train(k)
        return lda_handler.get_theta(), k

    def get_lda_relation(self):
        similarity = cosine_similarity(self.lda_distribution)
        return similarity.tolist()

    def get_member(self, index):
        for mem in self.members:
            if mem.get_index() == index:
                return mem
        raise IndexError

    def get_distances(self):
        distances = [[0.0 for j in range(len(self.methods))] for i in range(len(self.methods))]
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                distances[i][j] = (1.0 / abs(self.methods[i].get_index() - self.methods[j].get_index()))
                distances[j][i] = (1.0 / abs(self.methods[i].get_index() - self.methods[j].get_index()))
        return distances

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
                    cdm1 = self.methods[j].get_inner_invocations().count(self.methods[i].get_name()) / (
                            len(self.methods[j].get_inner_invocations()) + len(
                        self.methods[j].get_external_invocations()))
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

    def get_CSMs(self):
        # similarity = cosine_similarity_cache(self.distribution)
        similarity = cosine_similarity(self.distribution)
        return similarity.tolist()

    def get_summary_similarity(self):
        # glove_model = word2vec_model
        # try:
        #     texts = [camel_to_space(mem.get_gpt_text().split('.')[0]) for mem in self.methods]
        #     processed_sentences = [[word for word in sentence.lower().split() if word in glove_model] for sentence in
        #                            texts]
        #     vectors = [sum([glove_model[word] for word in sentence]) for sentence in processed_sentences]
        #     similarity = cosine_similarity(vectors)
        #     for i in range(len(similarity)):
        #         for j in range(len(similarity[i])):
        #             similarity[i][j] = min(1.0, max(0.0, (similarity[i][j] - 0.9) * 10))
        #     return similarity
        # except AttributeError:
        #     print("gpt text error at %s" % (str(self.god_class.get_index())))
        #     texts = []
        #     for mem in self.methods:
        #         try:
        #             texts.append(camel_to_space(mem.get_gpt_text().split('.')[0]))
        #         except AttributeError:
        #             texts.append('none')
        #     processed_sentences = [[word for word in sentence.lower().split() if word in glove_model] for sentence in
        #                            texts]
        #     vectors = [sum([glove_model[word] for word in sentence]) for sentence in processed_sentences]
        #     similarity = cosine_similarity(vectors)
        #     for i in range(len(similarity)):
        #         for j in range(len(similarity[i])):
        #             similarity[i][j] = min(1.0, max(0.0, (similarity[i][j] - 0.9) * 10))
        #     return similarity
        model = bgemodel
        # texts = [camel_to_space(mem.get_gpt_text().split('.')[0]) for mem in self.methods]
        # texts = [mem.get_code_summary() for mem in self.methods]
        texts = [mem.get_gpt_text() for mem in self.methods]
        vectors = model.encode(texts)
        self.timer.record_past('summary encode')
        similarity = cosine_similarity(vectors)
        for i in range(len(similarity)):
            for j in range(len(similarity[i])):
                similarity[i][j] = min(1.0, max(0.0, (similarity[i][j] - 0.667) * 3))
        return similarity

    # def split_camel_case(self, string):
    #     pattern = r'(?<!^)(?=[A-Z])'
    #     words = re.sub(pattern, ' ', string).split()
    #     return words
    #
    # def get_CSMs(self):
    #     documents = []
    #     for i in range(len(self.methods)):
    #         documents.append(re.sub(r"\W+", " ", self.methods[i].get_full_text()))
    #     o_texts = [[word for word in document.split()] for document in documents]
    #     texts = []
    #     for document in o_texts:
    #         new_document = []
    #         for word in document:
    #             new_document += self.split_camel_case(word)
    #         texts.append([w.lower() for w in new_document])
    #     dictionary = corpora.Dictionary(texts)
    #     corpus = [dictionary.doc2bow(text) for text in texts]
    #     lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=10)
    #     corpus_lsi = lsi[corpus]
    #     index = similarities.MatrixSimilarity(corpus_lsi)
    #     sims = index[corpus_lsi].tolist()
    #     for line in sims:
    #         for i in range(len(line)):
    #             line[i] = max(0.0, line[i])
    #     return sims

    # def get_externals(self):
    #     externals = [[0.0 for j in range(len(self.methods))] for i in range(len(self.methods))]
    #     for i in range(len(self.methods)):
    #         member_external_invocations1 = [c.split('+')[0] for c in self.members[i].get_external_invocations()]
    #         for j in range(i + 1, len(self.methods)):
    #             member_external_invocations2 = [c.split('+')[0] for c in self.members[j].get_external_invocations()]
    #             set1 = set(member_external_invocations1)
    #             set2 = set(member_external_invocations2)
    #             union = set1.union(set2)
    #             intersection = set1.intersection(set2)
    #             if len(union) > 0:
    #                 externals[i][j] = len(intersection) / len(union)
    #                 externals[j][i] = len(intersection) / len(union)
    #     return externals

    # def k_means_get_groups(self):
    #     kmeans = KMeans(n_clusters=max(self.k, 2), random_state=0, n_init=10)
    #     kmeans.fit_transform(self.distribution)
    #     clustering_result = kmeans.labels_
    #     groups: list[Group] = []
    #     for i in range(max(clustering_result) + 1):
    #         groups.append(Group())
    #     for i in range(len(clustering_result)):
    #         if self.members[i].is_method():
    #             groups[clustering_result[i]].append(self.members[i])
    #     return groups

    def get_init_groups(self):
        groups = []
        i = 2
        while i < len(self.methods):
            new_group = Group()
            new_group.append(self.methods[i - 2])
            new_group.append(self.methods[i - 1])
            new_group.append(self.methods[i])
            groups.append(new_group)
            i += 3
        if i < len(self.methods) + 2:
            new_group = Group()
            for j in range(i - 2, len(self.methods)):
                new_group.append(self.methods[j])
            groups.append(new_group)
        return groups

    def generate_graph(self):
        nodes = self.methods[:]
        edges = []
        weights = []
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                edges.append((self.methods[i], self.methods[j]))
                weights.append(self.get_method_coupling(i, j)[0])
        n = len(self.methods)
        while True:
            graph = nx.Graph()
            max_n_indexes = sorted(range(len(weights)), key=lambda x: weights[x])[-n:]
            for i in range(len(nodes)):
                graph.add_node(nodes[i])
            for i in max_n_indexes:
                graph.add_edge(edges[i][0], edges[i][1], weight=weights[i])
            chain_number = 0
            for component in nx.connected_components(graph):
                chain_number += 1
            if chain_number >= 2:
                break
            else:
                n -= 1
        return graph, edges, weights

    # def get_chain_number(self):
    #     chain_number = 0
    #     for component in nx.connected_components(self.chains):
    #         chain_number += 1
    #     return chain_number
    #
    # def graph_to_chains(self):
    #     # 二分查找满足得到2个链的阈值
    #     test_thresholds = [i * 0.05 for i in range(0, 300)]
    #     for threshold in test_thresholds:
    #         self.chains = self.graph.copy()
    #         for i in range(len(self.edges)):
    #             if threshold > self.chains.get_edge_data(self.edges[i][0], self.edges[i][1])['weight']:
    #                 self.chains.remove_edge(self.edges[i][0], self.edges[i][1])
    #         if self.chains.number_of_edges() <= len(self.methods) and self.get_chain_number() >= 2:
    #             return
    #     tqdm.write("failed to search threshold, use fixed")
    #     self.graph_to_chain()

    # def graph_to_chain(self):
    #     # 固定阈值
    #     threshold = 5.0 * sum(self.weight) / len(self.weight)
    #     # print("average weight = %s" % str(sum(self.weight) / len(self.weight)))
    #     self.chains = self.graph.copy()
    #     for i in range(len(self.edges)):
    #         if threshold > self.chains.get_edge_data(self.edges[i][0], self.edges[i][1])['weight']:
    #             self.chains.remove_edge(self.edges[i][0], self.edges[i][1])

    def chain_to_groups(self):
        groups = []
        for component in nx.connected_components(self.graph):
            new_group = Group()
            for member in component:
                new_group.append(member)
            groups.append(new_group)
        self.groups = groups

    def group_modify(self):
        self.remove_empty_group()
        move_threshold = 10
        last_move_mem = None
        while len(self.groups) > 2:
            members_to_be_move = []
            # 遍历类中所有方法，确定根据阈值需被移动的方法
            for mem in self.methods:
                for group in self.groups:
                    if mem in group:
                        if group.get_membership(mem) < move_threshold:
                            members_to_be_move.append(mem)
            # 若无方法符合移动条件，则终止
            if len(members_to_be_move) == 0:
                break
            # 对待移动method进行排序
            # members_to_be_move.sort(key=lambda x: self.get_members_group(x).get_membership(x))
            members_to_be_move.sort(key=lambda x: self.get_max_membership(x), reverse=True)
            move_flag = False
            for mem in members_to_be_move:
                sorted_groups = self.get_target_group(mem)
                if mem in sorted_groups[0] or mem == last_move_mem:
                    continue
                else:
                    for group in self.groups:
                        if mem in group:
                            group.remove(mem)
                        if group == sorted_groups[0]:
                            group.append(mem)
                    move_flag = True
                    last_move_mem = mem
                    break
            self.remove_empty_group()
            # 若未作移动则终止
            if not move_flag:
                break

    def get_target_group(self, member: Member):
        groups = self.groups[:]
        groups.sort(key=lambda x: self.get_score(member, x), reverse=True)
        return groups

    def get_method_coupling(self, index1: int, index2: int):
        """two index of self.methods """
        # 2.0, 1.5, 1.5, 0.7
        w = self.parameters_weight
        coupling = w[0] * self.distances[index1][index2] + w[1] * self.SSMs[index1][index2] + w[1] * self.CDMs[index1][
            index2] + w[2] * self.CSMs[index1][index2] + w[3] * self.code_summary_similarity[index1][index2] + w[4] * \
                   self.lda_relation[index1][
                       index2]  # + 0.0 * self.externals[index1][index2] #  + 0.0 * self.sentence_relations[index1][index2]
        return coupling, (coupling - 0.5 * w[0] * self.distances[index1][index2])

    def get_score(self, member: Member, group: Group):
        score = 0
        group_len = len(group) - int(member in group)
        if group_len != 0:
            i = self.methods.index(member)
            for mem in group.members:
                j = self.methods.index(mem)
                score += self.get_method_coupling(i, j)[0] / group_len
        return score

    def get_max_membership(self, member: Member):
        max_membership = 0.0
        for group in self.groups:
            max_membership = max(max_membership, self.get_score(member, group))
        return max_membership

    def get_members_group(self, member: Member):
        for group in self.groups:
            if member in group:
                return group
        raise KeyError

    def merge_group(self):
        while len(self.groups) > 2:
            pairs = []
            for i in range(len(self.groups)):
                for j in range(i + 1, len(self.groups)):
                    pairs.append((i, j))
            memberships = []
            for pair in pairs:
                memberships.append(self.compute_group_relation(pair))
            merge_pair = pairs[memberships.index(max(memberships))]
            self.groups[merge_pair[0]] = self.groups[merge_pair[0]] + self.groups[merge_pair[1]]
            self.groups.pop(merge_pair[1])

    def compute_group_relation(self, pair):
        group1 = self.groups[pair[0]]
        group2 = self.groups[pair[1]]
        r_sum = 0.0
        for mem in group1.members:
            r_sum += self.get_score(mem, group2)
        return r_sum / len(group1)

    def draw_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=False)
        # edge_labels = nx.get_edge_attributes(self.graph, "weight")

    def set_parameters_weight(self, l, s, t, r, lda=0.0):
        self.parameters_weight = (l, s, t, r, lda)

    def handel_fields(self, use_position=True):
        fields = self.get_fields()
        single_fields = []
        for field in fields:
            visit_times = []
            for i in range(len(self.groups)):
                visit_time = 0
                for method in self.groups[i]:
                    visit_time += method.get_visit_times(field.get_name())
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
                            score += 1.0 / abs(member.get_index() - field.get_index())
                        if member.is_method():
                            if member.get_index() > field.get_index():
                                score += 0.0
                                # score += 1.0 / abs(member.get_index() - field.get_index()) if abs(
                                #     member.get_index() - field.get_index()) < 10 else 0.0
                    distance_scores.append(score)
                self.groups[distance_scores.index(max(distance_scores))].append(field)
            else:
                self.groups[random.randint(0, len(self.groups) - 1)].append(field)


def camel_to_space(string: str):
    last_space = False
    last_upper = False
    new_string = ""
    for i in range(len(string)):
        if string[i] == ' ':
            new_string += string[i]
            last_space = True
            last_upper = False
        elif string[i].isupper():
            if (not last_upper) and (not last_space):
                new_string += " "
            if i + 1 < len(string) and string[i + 1].islower() and last_upper:
                new_string += " "
            new_string += string[i].lower()
            last_space = False
            last_upper = True
        elif string[i].islower():
            new_string += string[i]
            last_space = False
            last_upper = False
    return new_string


class ClusterDataContainer:

    def __init__(self, god_class, distribution, distances, SSMs, CDMs, CSMs, code_summary_similarity, timer):
        self.god_class = god_class
        self.distribution = distribution
        self.distances = distances
        self.SSMs = SSMs
        self.CDMs = CDMs
        self.CSMs = CSMs
        self.code_summary_similarity = code_summary_similarity
        self.timer = timer

    def get_index(self):
        return self.god_class.get_index()


class ClusterByStep(Cluster):

    def __init__(self, god_class):
        super().__init__(god_class)

    def topic_cluster_to_group(self):
        # kmeans = KMeans(n_clusters=max(self.k, 2), random_state=0, n_init=10)
        # kmeans.fit_transform(self.lda_distribution)
        # clustering_result = kmeans.labels_
        # groups: list[Group] = []
        # for i in range(max(clustering_result) + 1):
        #     groups.append(Group())
        # for i in range(len(clustering_result)):
        #     if self.members[i].is_method():
        #         groups[clustering_result[i]].append(self.members[i])
        # self.groups = groups
        groups: list[Group] = []
        for i in range(self.k):
            groups.append(Group())
        for i in range(len(self.lda_distribution)):
            if self.members[i].is_method():
                g = self.lda_distribution[i].index(max(self.lda_distribution[i]))
                groups[g].append(self.members[i])
        self.groups = groups

    def split_group_by_position(self):
        self.remove_empty_group()
        new_groups = []
        for j in range(len(self.groups)):
            group: Group = self.groups.pop()
            new_group = Group()
            for i in range(len(group)):
                new_member_index = group.indexes[i]
                if len(new_group) == 0:
                    new_group.append(self.get_method_by_index(new_member_index))
                    last_member_index = new_member_index
                    continue
                if self.is_adjacent(last_member_index, new_member_index):
                    new_group.append(self.get_method_by_index(new_member_index))
                else:
                    new_groups.append(new_group)
                    new_group = Group()
                    new_group.append(self.get_method_by_index(new_member_index))
                last_member_index = new_member_index
            new_groups.append(new_group)
        self.groups = new_groups

    def get_method_by_index(self, index):
        for mem in self.members:
            if mem.get_index() == index:
                return mem
        raise IndexError

    def is_adjacent(self, index1: int, index2: int, distance=1):
        if abs(index1 - index2) <= distance:
            return True
        middle_indexes = range(min(index1, index2) + 1, max(index1, index2))
        middle_method_num = 0
        for index in middle_indexes:
            if self.get_method_by_index(index).is_method():
                middle_method_num += 1
        if middle_method_num >= distance:
            return False
        else:
            return True

    def merge_split_group(self, l, s, t, r, lda):
        self.set_parameters_weight(l, s, t, r, lda)
        self.graph, self.edges, self.weight = self.generate_group_graph()
        self.group_graph_to_groups()
        self.merge_group()

    def generate_group_graph(self):
        self.remove_empty_group()
        nodes = self.groups[:]
        edges = []
        weights = []
        for i in range(len(self.groups)):
            for j in range(i + 1, len(self.groups)):
                edges.append((self.groups[i], self.groups[j]))
                weights.append(self.get_group_coupling(i, j))
        n = len(self.groups)
        while True:
            graph = nx.Graph()
            max_n_indexes = sorted(range(len(weights)), key=lambda x: weights[x])[-n:]
            for i in range(len(nodes)):
                graph.add_node(nodes[i])
            for i in max_n_indexes:
                graph.add_edge(edges[i][0], edges[i][1], weight=weights[i])
            chain_number = 0
            for component in nx.connected_components(graph):
                chain_number += 1
            if chain_number >= 2:
                break
            else:
                n -= 1
        return graph, edges, weights

    def get_group_coupling(self, index1, index2):
        group1 = self.groups[index1]
        group2 = self.groups[index2]
        l1 = len(group1)
        l2 = len(group2)
        r_sum = 0.0
        for i in range(l1):
            method1_index = self.methods.index(group1[i])
            for j in range(l2):
                method2_index = self.methods.index(group2[j])
                r_sum += self.get_method_coupling(method1_index, method2_index)[0]
        result = r_sum / (l1 * l2)
        return result

    def group_graph_to_groups(self):
        groups = []
        for component in nx.connected_components(self.graph):
            new_group = Group()
            for group in component:
                new_group += group
            groups.append(new_group)
        self.groups = groups


class ClusterSeparate(ClusterByStep):

    def __init__(self, god_class):
        super().__init__(god_class)
        self.index_chart = self.get_index_chart()
        self.use_similarity = [1, 1, 1, 1]
        self.thresholds = None
        self.algo_front = True
        self.algo_back = True

    def set_use_similarity(self, use: list[int]):
        self.use_similarity = use

    def set_use_front_algorithm(self, algo):
        self.algo_front = algo

    def set_use_back_algorithm(self, algo):
        self.algo_back = algo

    def get_index_chart(self):
        chart = [len(self.members)]
        for i in range(len(self.members)):
            if self.members[i] in self.methods:
                chart.append(self.methods.index(self.members[i]))
            else:
                chart.append(len(self.members))
        return chart

    def separate_to_group(self):
        if self.algo_front:
            self.generate_separate_graph()
            self.graph_to_2_chain()
        else:
            self.generate_nodes()
        self.chain_to_groups()

    def generate_nodes(self):
        nodes = self.methods[:]
        graph = nx.Graph()
        for i in range(len(nodes)):
            graph.add_node(nodes[i])
        self.graph = graph
        self.edges = []
        self.weight = []

    def generate_separate_graph(self):
        nodes = self.methods[:]
        edges = []
        weights = []
        n = len(self.methods)
        graph = nx.Graph()
        for i in range(len(nodes)):
            graph.add_node(nodes[i])
        self.thresholds = self.get_top(5, 8)
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                if self.algo_back:
                    weight = self.get_weight1(i, j)
                else:
                    weight = self.get_weight_all(i, j)
                if weight > (0.0 if self.algo_back else 1.0):
                    edges.append((self.methods[i], self.methods[j]))
                    weights.append(weight)
        for i in range(len(edges)):
            graph.add_edge(edges[i][0], edges[i][1], weight=weights[i])
        self.graph = graph
        self.edges = edges
        self.weight = weights

    def get_weight1(self, index1: int, index2: int):
        weight = 0

        # if self.distances[index1][index2] >= 0.9:
        #     weight += self.use_similarity[0] * 0.0
        # elif self.distances[index1][index2] >= 0.3:
        #     weight += self.use_similarity[0] * 0.0
        # else:
        #     weight += self.use_similarity[0] * 0.0
        #
        # if self.SSMs[index1][index2] + 0 * self.CDMs[index1][index2] >= 0.6:
        #     weight += self.use_similarity[1] * 0.0
        # else:
        #     weight += self.use_similarity[1] * 0.0

        # if self.CSMs[index1][index2] >= 0.8:
        #     weight += self.use_similarity[2] * 0.1
        # elif self.CSMs[index1][index2] >= 0.7:
        #     weight += self.use_similarity[2] * 0.0
        # else:
        #     weight += self.use_similarity[2] * 0.0
        #
        # if self.code_summary_similarity[index1][index2] >= 0.8:
        #     weight += self.use_similarity[3] * 0.1
        # elif self.code_summary_similarity[index1][index2] >= 0.75:
        #     weight += self.use_similarity[3] * 0.1
        # else:
        #     weight += self.use_similarity[3] * 0.0

        if self.CSMs[index1][index2] + self.code_summary_similarity[index1][index2] >= self.thresholds[1]:
            weight += 0.5 * (self.CSMs[index1][index2] + self.code_summary_similarity[index1][index2]) * \
                      self.use_similarity[2]
        # if self.code_summary_similarity[index1][index2] >= self.thresholds[2]:
        #     weight += self.code_summary_similarity[index1][index2] * self.use_similarity[3]

        return weight

    def get_weight_all(self, index1: int, index2: int):
        weight = self.distances[index1][index2] * self.use_similarity[0] + (
                self.SSMs[index1][index2] + self.CDMs[index1][index2]) * self.use_similarity[1] + 0.5 * (
                         self.CSMs[index1][index2] + self.code_summary_similarity[index1][index2]) * \
                 self.use_similarity[2]
        return weight

    def get_top(self, m, i):
        MS = []
        IS = []
        # SS = []
        for i_ in range(len(self.methods)):
            for j in range(i_ + 1, len(self.methods)):
                MS.append(self.SSMs[i_][j] + self.CDMs[i_][j])
                IS.append(self.CSMs[i_][j] + self.code_summary_similarity[i_][j])
                # SS.append(self.code_summary_similarity[i_][j])
        MS.sort(reverse=True)
        MS_result = MS[math.ceil(len(MS) * m / 100)]
        IS.sort(reverse=True)
        IS_result = IS[math.ceil(len(IS) * i / 100)]
        # SS.sort(reverse=True)
        # SS_result = SS[math.ceil(len(SS) * s / 100)]
        return MS_result, IS_result

    def graph_to_2_chain(self):
        if len(self.weight) == 0:
            return
        threshold = min(self.weight)
        while True:
            if self.get_chain_number() >= 2 or threshold > 2:
                return
            edges = self.edges[:]
            weights = self.weight[:]
            for i in range(len(edges)):
                if self.graph.get_edge_data(edges[i][0], edges[i][1])['weight'] <= threshold:
                    self.edges.remove(edges[i])
                    self.weight.remove(self.graph.get_edge_data(edges[i][0], edges[i][1])['weight'])
                    self.graph.remove_edge(edges[i][0], edges[i][1])
            threshold += 0.1

    def get_chain_number(self):
        chain_number = 0
        for component in nx.connected_components(self.graph):
            if len(component) > (2 if self.algo_back else 2):
                chain_number += 1
        return chain_number

    def split_group_by_position(self):
        if not self.use_similarity[0] == 0:
            super().split_group_by_position()

    def merge_adjacent_single_method(self):
        if self.use_similarity[0] == 0 or len(self.groups) >= len(self.methods):
            return
        new_groups = []
        self.groups.sort(key=lambda x: x.get_average_index())
        single_flag = False
        for g in self.groups:
            if len(g) == 1:
                if single_flag:
                    new_groups[-1] = new_groups[-1] + g
                else:
                    new_groups.append(g)
                    single_flag = True
            else:
                new_groups.append(g)
                single_flag = False
        self.groups = new_groups

    def merge_separate_group(self):
        self.thresholds = self.get_top(5, 8)
        while len(self.groups) > 2:
            weight_dict = {}
            for i in range(len(self.groups)):
                for j in range(len(self.groups)):
                    if i != j:
                        if self.algo_front or self.algo_back:
                            total_weight, num1, num2 = self.get_group_weight(self.groups[i], self.groups[j])
                        else:
                            total_weight = self.get_group_weight2(self.groups[i], self.groups[j])
                        weight_dict[total_weight / (len(self.groups[i]) * len(self.groups[j]))] = (i, j)
            merge_pair = self.find_most_similar_groups(weight_dict)
            # if self.stop_merge_group_flag(list(weight_dict.keys())):
            #     break
            if merge_pair is None:
                break
            self.groups[merge_pair[0]] = self.groups[merge_pair[0]] + self.groups[merge_pair[1]]
            self.groups.pop(merge_pair[1])

    def find_most_similar_groups(self, weight_dict: dict):
        pairs = [v for k, v in sorted(weight_dict.items(), reverse=True)]
        for pair in pairs:
            size_of_new_group = len(self.groups[pair[0]]) + len(self.groups[pair[1]])
            if size_of_new_group < 0.85 * len(self.methods) and len(self.methods) - size_of_new_group > 3:
                return pair
        return None

    def stop_merge_group_flag(self, weights):
        for group in self.groups:
            if len(group) <= 10:
                return False
        if max(weights) >= 0.1 * 2 / 20:
            return False
        return True

    def get_group_weight(self, group1: Group, group2: Group):
        total_weight = 0.0
        set1 = set()
        set2 = set()
        for i in group1.indexes:
            for j in group2.indexes:
                weight = self.get_weight2(self.index_chart[i], self.index_chart[j])
                if weight > 0.0:
                    set1.add(i)
                    set2.add(j)
                    total_weight += weight
        return total_weight, len(set1), len(set2)

    def get_weight2(self, index1: int, index2: int):
        weight = 0.0
        # 80%->0.1, 70->0.05
        # if self.distances[index1][index2] >= 0.9:
        #     weight += self.use_similarity[0] * 0.1
        # elif self.distances[index1][index2] >= 0.4:
        #     weight += self.use_similarity[0] * 0.03
        # elif self.distances[index1][index2] >= 0.3:
        #     weight += self.use_similarity[0] * 0.02
        # else:
        #     weight += self.use_similarity[0] * 0.0
        #
        # if self.SSMs[index1][index2] + self.CDMs[index1][index2] >= 0.55:
        #     weight += self.use_similarity[1] * 0.1
        # else:
        #     weight += self.use_similarity[1] * 0.0
        #
        # if self.CDMs[index1][index2] >= 0.1:
        #     weight += self.use_similarity[1] * 0.0
        # else:
        #     weight += self.use_similarity[1] * 0.0
        #
        # if self.CSMs[index1][index2] >= 0.85:
        #     weight += self.use_similarity[2] * 0.15
        # elif self.CSMs[index1][index2] >= 0.80:
        #     # weight += 0.1 + 0.05 * (self.CSMs[index1][index2] - 0.75) / 0.05
        #     weight += self.use_similarity[2] * 0.1
        # elif self.CSMs[index1][index2] >= 0.75:
        #     # weight += 0.05 + 0.05 * (self.CSMs[index1][index2] - 0.7) / 0.05
        #     weight += self.use_similarity[2] * 0.05
        # else:
        #     weight += 0.0
        #
        # if self.code_summary_similarity[index1][index2] >= 0.85:
        #     weight += self.use_similarity[3] * 0.2
        # elif self.code_summary_similarity[index1][index2] >= 0.80:
        #     # weight += self.use_similarity[3] * (0.1 + 0.1 * (self.code_summary_similarity[index1][index2] - 0.75) / 0.05)
        #     weight += self.use_similarity[3] * 0.1
        # elif self.code_summary_similarity[index1][index2] >= 0.75:
        #     # weight += self.use_similarity[3] * (0.04 + 0.06 * (self.code_summary_similarity[index1][index2] - 0.7) / 0.05)
        #     weight += self.use_similarity[3] * 0.04
        # else:
        #     weight += self.use_similarity[3] * 0.0

        if self.distances[index1][index2] >= 0.3:
            weight += self.distances[index1][index2] * self.use_similarity[0]
        if self.SSMs[index1][index2] + self.CDMs[index1][index2] >= self.thresholds[0]:
            weight += (self.SSMs[index1][index2] + self.CDMs[index1][index2]) * self.use_similarity[1]
        if self.CSMs[index1][index2] + self.code_summary_similarity[index1][index2] >= self.thresholds[1]:
            weight += 0.5 * (self.CSMs[index1][index2] + self.code_summary_similarity[index1][index2]) * \
                      self.use_similarity[2]
        # if self.code_summary_similarity[index1][index2] >= self.thresholds[2]:
        #     weight += self.code_summary_similarity[index1][index2] * self.use_similarity[3]

        return weight

    def get_group_weight2(self, group1: Group, group2: Group):
        total_weight = 0.0
        set1 = set()
        set2 = set()
        for i in group1.indexes:
            for j in group2.indexes:
                weight = self.get_weight_all(self.index_chart[i], self.index_chart[j])
                total_weight += weight
        return total_weight
