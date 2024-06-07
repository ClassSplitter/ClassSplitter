import math
import matplotlib.pyplot as plt
import pandas as pd
import re
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List
from sentence_transformers import SentenceTransformer
from god_class import GodClass
from timer import Timer
from pandas.core.frame import DataFrame
import numpy as np


st5model = None
def awake_st5model(model_name: str):
    global st5model
    if st5model is None:
        st5model = SentenceTransformer(model_name)
        # st5model = SentenceTransformer("krlvi/sentence-t5-base-nlpl-code_search_net")
        # st5model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# bgemodel = SentenceTransformer('BAAI/bge-base-en')


class Member:

    def __init__(self, data: DataFrame):
        self.data: DataFrame = data

    def get_index(self) -> int:
        return int(self.data.at[self.data.iloc[0].name, 'index'])
    
    def get_indexes(self) -> list[int]:
        return [int(self.data.at[self.data.iloc[0].name, 'index'])]

    def get_name(self) -> str:
        return str(self.data.at[self.data.iloc[0].name, 'name'])
    
    def get_members(self):
        return [self]

    def is_method(self) -> bool:
        if self.data.at[self.data.iloc[0].name, 'type'] == 'Method':
            return True
        else:
            return False

    def is_field(self) -> bool:
        if self.data.at[self.data.iloc[0].name, 'type'] == 'Field':
            return True
        else:
            return False

    # def is_moved(self) -> bool:
    #     if self.data.at[self.data.iloc[0].name, 'moved']:
    #         return True
    #     else:
    #         return False

    def is_removed(self) -> bool:
        if self.data.at[self.data.iloc[0].name, 'removed']:
            return True
        else:
            return False

    def get_document(self) -> str:
        return str(self.data.at[self.data.iloc[0].name, 'document'])

    def get_inner_invocations(self) -> list[str]:
        if str(self.data.at[self.data.iloc[0].name, 'inner invocations']) == '':
            return []
        else:
            return str(self.data.at[self.data.iloc[0].name, 'inner invocations']).split()

    def get_external_invocations(self) -> list[str]:
        if str(self.data.at[self.data.iloc[0].name, 'external invocations']) == '':
            return []
        else:
            return str(self.data.at[self.data.iloc[0].name, 'external invocations']).split()

    def get_visits(self) -> list[str]:
        if str(self.data.at[self.data.iloc[0].name, 'visits']) == 'nan':
            return []
        else:
            return str(self.data.at[self.data.iloc[0].name, 'visits']).split()

    def get_full_text(self) -> str:
        return str(self.data.at[self.data.iloc[0].name, 'full text'])
    
    def get_lines(self) -> int:
        return int(self.data.at[self.data.iloc[0].name, 'lines'])

    def get_visit_times(self, name: str) -> int:
        if str(self.data.at[self.data.iloc[0].name, 'visits']) == 'nan':
            return 0
        visit_full_names = str(self.data.at[self.data.iloc[0].name, 'visits']).split()
        visits = [string.split('+')[1] for string in visit_full_names]
        return visits.count(name)
    
    def get_code_summary(self) -> str:
        return str(self.data.at[self.data.iloc[0].name, 'code summary'])


class Unit:
    '''不可分割的methods集合'''

    def __init__(self, index: int, methods: list[Member]):
        self.index: int = index
        self.methods: list[Member] = methods
        self.related_field_names: list[str] = []
        # assert all methods are neighboring
        for i in range(len(self.methods) - 1):
            assert self.methods[i + 1].get_index() - self.methods[i].get_index() == 1

    def __contains__(self, item: Member | int) -> bool:
        if isinstance(item, Member):
            if item in self.methods:
                return True
        elif isinstance(item, int):
            if item in self.get_indexes():
                return True
        return False

    def __len__(self) -> int:
        return len(self.methods)
    
    def get_index(self) -> int:
        return self.index
    
    def get_members(self) -> list[Member]:
        return self.methods

    def get_names(self) -> list[str]:
        names: list = []
        for member in self.methods:
            names.append(member.get_name())
        return names
    
    def is_field(self) -> bool:
        return False
    
    def is_method(self) -> bool:
        return True
    
    def get_indexes(self) -> list[int]:
        indexes: list = []
        for method in self.methods:
            indexes.append(method.get_index())
        return indexes
    
    def get_lines(self) -> int:
        lines = 0
        for mem in self.methods:
            lines += mem.get_lines()
        return lines

    def get_average_index(self):
        index_sum = 0.0
        for mem in self.methods:
            index_sum += mem.get_index()
        return index_sum / len(self.methods)
    
    def get_visits(self) -> list[str]:
        visits: list[str] = []
        for mem in self.methods:
            visits += mem.get_visits()
        return list(set(visits))
    
    def get_inner_invocations(self):
        invotations:list[str] = []
        for mem in self.methods:
            invotations += mem.get_inner_invocations()
        return invotations
    
    def get_external_invocations(self) -> list[str]:
        invotations: list[str] = []
        for mem in self.methods:
            invotations += mem.get_external_invocations()
        return invotations
    
    def get_visit_times(self, name: str) -> int:
        count: int = 0
        for method in self.methods:
            count += method.get_visit_times(name)
        return count

    def get_moved_number(self):
        moved_number = 0
        for mem in self.methods:
            if mem.is_moved():
                moved_number += 1
        return moved_number

    def get_removed_number(self):
        removed_number = 0
        for mem in self.methods:
            if mem.is_removed():
                removed_number += 1
        return removed_number

    def get_moved_percentage(self):
        return self.get_moved_number() / len(self.methods)
    

class Group:
    '''Unit | Member的集合，其中方法若未特殊标明，均基于Unit计算'''

    def __init__(self):
        self.units: list[Unit | Member] = []

    def __add__(self, other):
        self.units += other.units
        self.sort()
        return self

    def __contains__(self, item: int | Member | Unit) -> bool:
        '''int item should be member index, not unit index'''
        if isinstance(item, Member) or isinstance(item, Unit):
            if item in self.units:
                return True
        elif isinstance(item, int):
            if item in self.get_indexes():
                return True
        return False

    def __len__(self):
        return len(self.units)

    def __getitem__(self, index) -> Unit | Member:
        return self.units[index]
    
    def sort(self):
        self.units.sort(key=lambda x: x.get_index())

    def append(self, unit: Unit):
        self.units.append(unit)
        self.sort()
    
    def get_units(self) -> list[Unit]:
        return self.units
    
    def get_indexes(self) -> list[int]:
        indexes = []
        for unit in self.units:
            indexes.append(unit.get_index())
        return indexes
    
    def get_members(self) -> list[Member]:
        '''based on members'''
        memebrs: list = []
        for unit in self.units:
            memebrs += unit.get_members()
        return memebrs
    
    def get_member_indexes(self) -> list[int]:
        '''based on members'''
        indexes = []
        for unit in self.units:
            indexes += unit.get_indexes()
        return indexes

    def get_names(self) -> list[str]:
        '''based on members'''
        names = []
        for unit in self.units:
            names += unit.get_names()
        return names
    
    def get_method_number(self) -> int:
        '''based on members'''
        return len(self.get_members())
    
    def get_lines(self) -> int:
        lines = 0
        for unit in self.units:
            lines += unit.get_lines()
        return lines
    
    def get_visits(self) -> list[str]:
        visits: list[str] = []
        for mem in self.units:
            visits += mem.get_visits()
        return list(set(visits))

    def get_average_index(self):
        '''based on members'''
        index_sum = 0.0
        for mem in self.get_members():
            index_sum += mem.get_index()
        return index_sum / len(self.get_members())

    def get_moved_number(self):
        '''based on members'''
        moved_number = 0
        for mem in self.get_members():
            if mem.is_moved():
                moved_number += 1
        return moved_number

    def get_removed_number(self):
        '''based on members'''
        removed_number = 0
        for mem in self.get_members():
            if mem.is_removed():
                removed_number += 1
        return removed_number

    def get_moved_percentage(self) -> float:
        '''based on members'''
        return self.get_moved_number() / len(self.get_members())
    
    def split_by_position(self) -> List:
        self.sort()
        new_groups = []
        new_group = Group()
        for u in self.units:
            new_member_index = u.get_index()
            if len(new_group) == 0:
                new_group.append(u)
                last_member_index = new_member_index
                continue
            if abs(last_member_index - new_member_index) <= 1:
                new_group.append(u)
            else:
                new_groups.append(new_group)
                new_group = Group()
                new_group.append(u)
            last_member_index = new_member_index
        new_groups.append(new_group)
        return new_groups


class GroupInterface:
    def __init__(self, god_class: GodClass):
        self.groups: list[Group] = []
        self.figure_number = 0
        self.god_class = god_class
        self.members: List[Member] = self.get_members()
        self.methods: List[Member] = self.get_methods()
        self.fields: List[Member] = self.get_fields()
    
    def get_id(self) -> int:
        return self.god_class.get_last_index()

    def get_members(self) -> List[Member]:
        members = []
        for i in range(len(self.god_class.data_frame)):
            member = Member(pd.DataFrame(self.god_class.data_frame.iloc[i]).T)
            members.append(member)
        return members

    def get_methods(self) -> List[Member]:
        methods = []
        for mem in self.members:
            if mem.is_method():
                methods.append(mem)
        methods.sort(key=lambda x: x.get_index())
        return methods

    def get_fields(self) -> List[Member]:
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

    def get_mojofm(self, field=False):
        self.group_to_rsf()
        self.example_to_rsf(field=field)
        f = os.popen('java mojo.MoJo group.rsf example.rsf -fm')
        r_str = f.read()
        result = float(r_str.strip())
        f.close()
        return max(0.0, (result - 50.0) / 50.0)

    def group_to_rsf(self):
        str_builder = ""
        for i in range(len(self.groups)):
            c_name = 'c' + str(i)
            for mem in self.groups[i].get_members():
                str_builder += 'contain ' + c_name + ' ' + str(mem.get_index()) + '\n'
        f = open('group.rsf', 'w')
        f.write(str_builder)
        f.close()

    def example_to_rsf(self, field):
        str_builder = ""
        for i in range(len(self.members)):
            c_name = 'c' + str(self.members[i].is_removed())
            if self.members[i].is_method() or (self.members[i].is_field() and field):
                str_builder += 'contain ' + c_name + ' ' + str(self.members[i].get_index()) + '\n'
        f = open('example.rsf', 'w')
        f.write(str_builder)
        f.close()

    def get_field_mojofm(self):
        str_builder = ""
        for i in range(len(self.groups)):
            c_name = 'c' + str(i)
            for mem in self.groups[i].get_members():
                if mem.is_field():
                    str_builder += 'contain ' + c_name + ' ' + str(mem.get_index()) + '\n'
        if str_builder == "":
            # print("no field in class %s" % str(self.god_class.get_index()))
            return 0.5
        f = open('group.rsf', 'w')
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
        f = open('example.rsf', 'w')
        f.write(str_builder)
        f.close()
        f = os.popen('java mojo.MoJo group.rsf example.rsf -fm')
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

    def __init__(self, god_class: GodClass, model_name, cache_path):
        super().__init__(god_class)
        self.model_name = model_name
        if os.path.exists(cache_path):
            history_df = pd.read_json(cache_path)
        else:
            history_df = pd.DataFrame(columns=['id', 'distribution', 'PS', 'MS', 'CS', 'time'])
        if history_df['id'].isin([self.get_id()]).any():
            history_df_data = history_df[history_df['id'] == self.get_id()]
            assert len(history_df_data) == 1
            self.distribution = None
            self.PS = history_df_data['PS'].values[0]
            self.MS = history_df_data['MS'].values[0]
            self.CS = history_df_data['CS'].values[0]
            self.timer = Timer(history_df_data['time'].values[0])
        else:
            self.timer = Timer()
            self.timer.start_record('PS')
            self.PS = self.get_distances()
            self.timer.end_record('PS')
            self.timer.start_record('MS')
            self.MS = self.get_MS()
            self.timer.end_record('MS')
            self.timer.start_record('distribution')
            self.distribution = self.get_distribution()
            self.timer.end_record('distribution')
            self.timer.start_record('CS')
            self.CS = self.get_CS()
            self.timer.end_record('CS')
            history_df.loc[len(history_df)] = [self.get_id(), self.distribution, self.PS, self.MS, self.CS, self.timer.get_record()]
            history_df.to_json(cache_path)
        self.graph = None
        self.edges = None
        self.weight = None

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
    
    def get_MS(self):
        SSM = self.get_SSMs()
        CDM = self.get_CDMs()
        ms = [[0.0 for j in range(len(self.methods))] for i in range(len(self.methods))]
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                ms[i][j] = SSM[i][j] + CDM[i][j]
                ms[j][i] = SSM[j][i] + CDM[j][i]
        return ms

    def get_CS(self):
        similarity = cosine_similarity(self.distribution)
        return similarity.tolist()
    
    def get_distribution(self):
        awake_st5model(self.model_name)
        embeddings = st5model.encode([method.get_full_text() for method in self.methods])
        return embeddings

    def get_members_group(self, member: Member):
        for group in self.groups:
            if member in group:
                return group
        raise KeyError
    
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
    
    def graph_to_groups(self):
        groups = []
        for component in nx.connected_components(self.graph):
            new_group = Group()
            for member in component:
                new_group.append(member)
            groups.append(new_group)
        self.groups = groups


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


class ClusterSeparate(Cluster):

    def __init__(self, god_class: GodClass, 
                 model_name="krlvi/sentence-t5-base-nlpl-code_search_net", 
                 cache_path='cache'):
        super().__init__(god_class, model_name, cache_path)
        self.index_chart = self.get_index_chart()
        self.thresholds = None
        self.lines = self.get_lines()
        self.use_similarity = {'PS': 1, 'CS': 1, 'MS': 1}
        self.algo_front = True
        self.algo_back = True

    def set_use_similarity(self, use: dict[str: int]):
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
    
    def get_lines(self):
        lines = 0
        for mem in self.methods:
            lines += mem.get_lines()
        return lines

    def separate_to_group(self):
        if self.algo_front:
            self.generate_separate_graph()
            self.graph_to_2_chain()
        else:
            self.generate_nodes()
        self.graph_to_groups()

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
        if self.CS[index1][index2] >= self.thresholds[1]:
            weight += self.CS[index1][index2] * self.use_similarity['CS']
        return weight

    def get_weight_all(self, index1: int, index2: int):
        weight = self.PS[index1][index2] * self.use_similarity['PS'] + self.MS[index1][index2] * self.use_similarity['MS'] + self.CS[index1][index2] * self.use_similarity['CS']
        return weight

    def get_top(self, m, i):
        MS = []
        IS = []
        # SS = []
        for i_ in range(len(self.methods)):
            for j in range(i_ + 1, len(self.methods)):
                MS.append(self.MS[i_][j])
                IS.append(self.CS[i_][j])
        MS.sort(reverse=True)
        MS_result = MS[math.ceil(len(MS) * m / 100)]
        IS.sort(reverse=True)
        IS_result = IS[math.ceil(len(IS) * i / 100)]
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
        if self.use_similarity['PS'] == 0:
            return
        self.remove_empty_group()
        new_groups = []
        for g in self.groups:
            new_groups += g.split_by_position()
        self.groups = new_groups

    def merge_adjacent_single_method(self):
        if self.use_similarity['PS'] == 0 or len(self.groups) >= len(self.methods):
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
                        total_weight = self.get_group_weight(self.groups[i], self.groups[j])
                        weight_dict[(i, j)] = total_weight / (len(self.groups[i]) * len(self.groups[j]))
            merge_pair = self.find_most_similar_groups(weight_dict)
            if merge_pair is None:
                break
            self.groups[merge_pair[0]] = self.groups[merge_pair[0]] + self.groups[merge_pair[1]]
            self.groups.pop(merge_pair[1])

    def find_most_similar_groups(self, weight_dict: dict):
        pairs = sorted(weight_dict, key=weight_dict.get, reverse=True)
        # for pair in pairs:
        #     size_of_new_group = len(self.groups[pair[0]]) + len(self.groups[pair[1]])
        #     if size_of_new_group < 0.85 * len(self.methods) and len(self.methods) - size_of_new_group > 3:
        #         return pair
        for pair in pairs:
            new_group_method_number = len(self.groups[pair[0]]) + len(self.groups[pair[1]])
            if new_group_method_number > 0.85 * len(self.methods) and weight_dict[pair] < 0.2:
                continue
            new_group_line_number = self.groups[pair[0]].get_lines() + self.groups[pair[1]].get_lines()
            if new_group_line_number > 0.85 * self.lines and weight_dict[pair] < 0.2:
                continue
            if weight_dict[pair] < 0.05:
                continue
            return pair
        return None

    def get_group_weight(self, group1: Group, group2: Group):
        total_weight = 0.0
        for i in group1.get_indexes():
            for j in group2.get_indexes():
                weight = self.get_weight2(self.index_chart[i], self.index_chart[j])
                if weight > 0.0:
                    total_weight += weight
        return total_weight

    def get_weight2(self, index1: int, index2: int):
        weight = 0.0

        if self.PS[index1][index2] >= 0.3:
            weight += self.PS[index1][index2] * self.use_similarity['PS']
        if self.MS[index1][index2] >= self.thresholds[0]:
            weight += self.MS[index1][index2] * self.use_similarity['MS']
        if self.CS[index1][index2] >= self.thresholds[1]:
            weight += self.CS[index1][index2] * self.use_similarity['CS']

        return weight
    
    def handle_fields(self, use_position=True):
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
                    distance_scores.append(score)
                self.groups[distance_scores.index(max(distance_scores))].append(field)
            else:
                self.groups[0].append(field)


def remove_prefix(s: str):
    # 检查是否以'get'、'set'或'is'开头
    ls: str
    if s.startswith('get'):
        ls = s[len('get'):].lower()
    elif s.startswith('set'):
        ls = s[len('set'):].lower()
    elif s.startswith('is'):
        ls = s[len('is'):].lower()
    else:
        return False
    return re.sub(r'\(([^()]*)\)', "", ls)


class ClusterSeparateUnit(GroupInterface):

    def __init__(self, god_class: GodClass, cache_path: str, model_name='sentence-transformers/all-mpnet-base-v2'):
        super().__init__(god_class)
        # cache
        if os.path.exists(cache_path):
            history_df = pd.read_json(cache_path)
        else:
            history_df = pd.DataFrame(columns=['id', 'distribution', 'time'])
        if history_df['id'].isin([self.get_id()]).any():
            history_df_data = history_df[history_df['id'] == self.get_id()]
            assert len(history_df_data) == 1
            self.distribution = history_df_data['distribution'].values[0]
        else:
            self.timer = Timer()
            self.timer.start_record('distribution')
            self.distribution = self.get_distribution(model_name)
            self.timer.end_record('distribution')
            history_df.loc[len(history_df)] = [self.get_id(), self.distribution, self.timer.get_record()]
            history_df.to_json(cache_path)
        # Intermediate states
        self.units: list[Unit] = []  # only method, no field
        self.ps: np.ndarray
        self.ms: np.ndarray
        self.fs: np.ndarray
        self.link_weights: list[float] = []
        self.cluster_parameters: dict = {}
        self.use_similarity = {'PS': 1, 'MS': 1, 'FS': 1}
        self.graph: nx.Graph
        self.edges: List[tuple[int]]
        self.weight: List[float]
        self.lines = self.get_lines()
    
    def get_distribution(self, model_name: str):
        awake_st5model(model_name)
        embeddings = st5model.encode([method.get_full_text() for method in self.methods])
        return embeddings
    
    def get_lines(self):
        lines = 0
        for mem in self.methods:
            lines += mem.get_lines()
        return lines
    
    def heuristic_method_link(self) -> None:
        """step0 根据启发式规则将部分method进行绑定，规则详见should_merge_method()"""
        units: list[Unit] = []
        pointer = 0
        while pointer < len(self.methods):
            new_unit_size = 1
            while pointer + new_unit_size < len(self.methods):
                merge_flag = False
                for i in range(pointer, pointer + new_unit_size):
                    merge_flag = merge_flag | self.should_merge_method(self.methods[i], self.methods[pointer + new_unit_size])
                if merge_flag:
                    new_unit_size += 1
                else:
                    break
            new_unit = Unit(len(units), self.methods[pointer: pointer + new_unit_size])
            units.append(new_unit)
            pointer += new_unit_size
        self.units = units
        self.compute_link_weight()
    
    def should_merge_method(self, m1: Member, m2: Member) -> bool:
        '''判断m1与m2是否应在拆分过程中绑定'''
        # should be adjacent
        if abs(m1.get_index() - m2.get_index()) != 1:
            return False
        # get,set,is method should merge
        if (field_name1 := remove_prefix(m1.get_name())) and (field_name2 := remove_prefix(m2.get_name())):
            if field_name1 == field_name2 and self.has_field(field_name1):
                return True
        # single function call should merge
        if len(m1.get_inner_invocations()) == 1 and len(m1.get_external_invocations()) == 0:
            if m1.get_inner_invocations()[0] == m2.get_name():
                return True
        if len(m2.get_inner_invocations()) == 1 and len(m2.get_external_invocations()) == 0:
            if m2.get_inner_invocations()[0] == m1.get_name():
                return True
        
        return False

    def has_field(self, name: str) -> bool:
        '''the class contains field with that name'''
        for field in self.fields:
            if field.get_name().lower() == name.lower():
                return True
        return False
    
    def compute_link_weight(self) -> None:
        '''计算链中的关系值，分为语义和度量两部分相加'''
        self.ms: np.ndarray = np.array(self.get_SSMs()) + np.array(self.get_CDMs())
        self.fs: np.ndarray = np.array(self.get_CS())
        self.ps: np.ndarray = np.array(self.get_PS())
    
    def get_SSMs(self) -> list[list[float]]:
        '''unit对field访问关系矩阵'''
        ssms = [[0.0 for j in range(len(self.units))] for i in range(len(self.units))]
        for i in range(len(self.units)):
            for j in range(i + 1, len(self.units)):
                set1 = set(self.units[i].get_visits())
                set2 = set(self.units[j].get_visits())
                union = set1.union(set2)
                intersection = set1.intersection(set2)
                if len(union) > 0:
                    ssms[i][j] = len(intersection) / len(union)
                    ssms[j][i] = len(intersection) / len(union)
        return ssms

    def get_CDMs(self) -> list[list[float]]:
        '''unit间调用关系矩阵'''
        cdms = [[0.0 for j in range(len(self.units))] for i in range(len(self.units))]
        for i in range(len(self.units)):
            for j in range(i + 1, len(self.units)):
                if len(self.units[j].get_inner_invocations()) + len(self.units[j].get_external_invocations()) > 0:
                    cdm1 = sum([self.units[j].get_inner_invocations().count(name) for name in self.units[i].get_names()]) / (
                            len(self.units[j].get_inner_invocations()) + len(self.units[j].get_external_invocations()))
                else:
                    cdm1 = 0.0
                if len(self.units[i].get_inner_invocations()) + len(self.units[i].get_external_invocations()) > 0:
                    cdm2 = sum([self.units[i].get_inner_invocations().count(name) for name in self.units[j].get_names()]) / (
                            len(self.units[i].get_inner_invocations()) + len(self.units[i].get_external_invocations()))
                else:
                    cdm2 = 0.0
                cdms[i][j] = max(cdm1, cdm2)
                cdms[j][i] = max(cdm1, cdm2)
        return cdms
    
    def get_CS(self) -> list[float]:
        '''unit间文本相似度矩阵'''
        unit_distribution: List[List[float]] = [[] for _ in range(len(self.units))]
        for i in range(len(self.units)):
            values: List[List[float]] = []
            for j in range(len(self.methods)):
                if self.methods[j].get_index() in self.units[i]:
                    values.append(self.distribution[j])
            unit_distribution[i] = np.mean(values, axis=0).tolist()
        similarity: np.ndarray = cosine_similarity(unit_distribution)
        similarity_norm: np.ndarray = (similarity - np.min(similarity)) / (np.max(similarity) - np.min(similarity))
        return similarity_norm.tolist()
    
    def get_PS(self) -> list[list[float]]:
        distances = [[0.0 for j in range(len(self.units))] for i in range(len(self.units))]
        for i in range(len(self.units)):
            for j in range(i + 1, len(self.units)):
                distances[i][j] = (1.0 / abs(self.units[i].get_index() - self.units[j].get_index()))
                distances[j][i] = (1.0 / abs(self.units[i].get_index() - self.units[j].get_index()))
        return distances
    
    def separate_to_group(self):
        """step1 根据fs绘图，后将图写入group"""
        self.generate_separate_graph()
        self.graph_to_2_chain()
        self.graph_to_groups()

    def generate_separate_graph(self):
        nodes = self.units[:]
        edges = []
        weights = []
        graph = nx.Graph()
        for i in range(len(nodes)):
            graph.add_node(nodes[i])
        self.thresholds = self.get_top(5, 8)
        for i in range(len(self.units)):
            for j in range(i + 1, len(self.units)):
                weight = self.get_weight1(i, j)
                if weight > 0.0:
                    edges.append((self.units[i], self.units[j]))
                    weights.append(weight)
        for i in range(len(edges)):
            graph.add_edge(edges[i][0], edges[i][1], weight=weights[i])
        self.graph = graph
        self.edges = edges
        self.weight = weights
    
    def get_top(self, m, i):
        MS = []
        FS = []
        # SS = []
        for i_ in range(len(self.units)):
            for j in range(i_ + 1, len(self.units)):
                MS.append(self.ms[i_][j])
                FS.append(self.fs[i_][j])
        MS.sort(reverse=True)
        MS_result = MS[math.ceil(len(MS) * m / 100)]
        FS.sort(reverse=True)
        FS_result = FS[math.ceil(len(FS) * i / 100)]
        return MS_result, FS_result
    
    def get_weight1(self, index1: int, index2: int) -> float:
        weight = 0.0
        if self.fs[index1][index2] >= self.thresholds[1]:
            weight += self.fs[index1][index2] * self.use_similarity['FS']
        return weight
    
    def graph_to_2_chain(self):
        if len(self.weight) == 0:
            return
        threshold = min(self.weight)
        while True:
            if self.get_chain_number() >= 2 or threshold > 2:
                return
            edges = self.edges[:]
            for i in range(len(edges)):
                if self.graph.get_edge_data(edges[i][0], edges[i][1])['weight'] <= threshold:
                    self.edges.remove(edges[i])
                    self.weight.remove(self.graph.get_edge_data(edges[i][0], edges[i][1])['weight'])
                    self.graph.remove_edge(edges[i][0], edges[i][1])
            threshold += 0.1
    
    def get_chain_number(self):
        chain_number = 0
        for component in nx.connected_components(self.graph):
            if len(component) > 2:
                chain_number += 1
        return chain_number
    
    def graph_to_groups(self):
        groups = []
        for component in nx.connected_components(self.graph):
            new_group = Group()
            for member in component:
                new_group.append(member)
            groups.append(new_group)
        self.groups = groups
    
    def split_group_by_position(self):
        """step2 根据位置拆分不相邻group成员"""
        if self.use_similarity['PS'] == 0:
            return
        self.remove_empty_group()
        new_groups = []
        for g in self.groups:
            new_groups += g.split_by_position()
        self.groups = new_groups

    def merge_adjacent_single_method(self):
        """step3 将单unit的位置连续group聚合"""
        if self.use_similarity['PS'] == 0 or len(self.groups) >= len(self.methods):
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
        """step4 将单unit的位置连续group聚合"""
        self.thresholds = self.get_top(5, 8)
        while len(self.groups) > 2:
            weight_dict = {}
            for i in range(len(self.groups)):
                for j in range(len(self.groups)):
                    if i != j:
                        total_weight = self.get_group_weight(self.groups[i], self.groups[j])
                        weight_dict[(i, j)] = total_weight / (len(self.groups[i]) * len(self.groups[j]))
            merge_pair = self.find_most_similar_groups(weight_dict)
            if merge_pair is None:
                break
            self.groups[merge_pair[0]] = self.groups[merge_pair[0]] + self.groups[merge_pair[1]]
            self.groups.pop(merge_pair[1])
    
    def get_group_weight(self, group1: Group, group2: Group):
        total_weight = 0.0
        for i in group1.get_indexes():
            for j in group2.get_indexes():
                weight = self.get_weight2(i, j)
                if weight > 0.0:
                    total_weight += weight
        return total_weight

    def get_weight2(self, index1: int, index2: int):
        weight = 0.0

        if self.ps[index1][index2] >= 0.3:
            weight += self.ps[index1][index2] * self.use_similarity['PS']
        if self.ms[index1][index2] >= self.thresholds[0]:
            weight += self.ms[index1][index2] * self.use_similarity['MS']
        if self.fs[index1][index2] >= self.thresholds[1]:
            weight += self.fs[index1][index2] * self.use_similarity['FS']

        return weight

    def find_most_similar_groups(self, weight_dict: dict):
        pairs = sorted(weight_dict, key=weight_dict.get, reverse=True)
        for pair in pairs:
            new_group_method_number = len(self.groups[pair[0]]) + len(self.groups[pair[1]])
            if new_group_method_number > 0.85 * len(self.methods) and weight_dict[pair] < 0.2:
                continue
            new_group_line_number = self.groups[pair[0]].get_lines() + self.groups[pair[1]].get_lines()
            if new_group_line_number > 0.85 * self.lines and weight_dict[pair] < 0.2:
                continue
            if weight_dict[pair] < 0.05:
                continue
            return pair
        return None
    
    def handle_fields(self, use_position=True):
        """step5 处理field"""
        fields = self.get_fields()
        single_fields: List[Member] = []
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
                    distance_scores.append(score)
                self.groups[distance_scores.index(max(distance_scores))].append(field)
            else:
                self.groups[0].append(field)

