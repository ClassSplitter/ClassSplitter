import os
import re
import math
import pickle
import pandas as pd
import pandas.core.frame
import networkx as nx
import openai
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


openai.api_key = ''


st5model = SentenceTransformer("krlvi/sentence-t5-base-nlpl-code_search_net")
bgemodel = SentenceTransformer('BAAI/bge-base-en')


class GodClass:

    def __init__(self, data: pandas.core.frame.DataFrame):
        self.data = data
        self.data_path = str(data.at[self.data.iloc[0].name, 'data path'])
        self.data_frame: pandas.core.frame.DataFrame = \
            pd.read_csv(self.data_path, delimiter=',', header=0, encoding="MacRoman")

    def __len__(self):
        return len(self.data_frame)

    def get_gpt_respond(self):
        print("\n============================ getting %s gpt respond. ============================" % self.get_name())
        respond = ["" for i in range(len(self.data_frame))]
        for i in range(len(self.data_frame)):
            if self.data_frame.at[i, 'type'] == "Method":
                print("in %s getting %dth respond out of %d, respond:" % (self.get_name(), i, len(self.data_frame)))
                try:
                    content = f"Please give me a function summary in less than 3 sentences for the following source code:\n```java\n{self.data_frame.at[i, 'full text']}\n```"
                    messages = [
                        {"role": "user",
                         "content": content},
                    ]
                    response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=messages,
                        temperature=0,
                    )
                    respond[i] = response['choices'][0]['message']['content']
                except Exception as e:
                    print(e)
            else:
                respond[i] = self.data_frame.at[i, 'commit']
            print(respond[i])
        self.data_frame['code summary'] = respond

    def to_csv(self, filename=None):
        if filename is None:
            filename = self.data_path
        self.data_frame.to_csv(filename, index=False)

    def get_method_number(self):
        num = 0
        for i in range(len(self.data_frame)):
            if self.data_frame.at[i, 'type'] == "Method":
                num += 1
        return num

    def get_field_number(self):
        num = 0
        for i in range(len(self.data_frame)):
            if self.data_frame.at[i, 'type'] == "Field":
                num += 1
        return num

    def get_name(self):
        return str(self.data.at[self.data.iloc[0].name, 'origin class'])

    def get_url(self):
        return str(self.data.at[self.data.iloc[0].name, 'url'])


class Member:
    """a class member (field/method/member class)"""

    def __init__(self, data: pandas.core.frame.DataFrame):
        self.data = data

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

    def get_document(self):
        return str(self.data.at[self.data.iloc[0].name, 'document'])

    def get_code_summary(self):
        try:
            text = self.data.at[self.data.iloc[0].name, 'code summary']
        except KeyError:
            return None
        if text == '':
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
    """a group of class members"""

    def __init__(self):
        self.members: list[Member] = []
        self.indexes: list[int] = []

    def __add__(self, other):
        self.members += other.members
        self.indexes += other.indexes
        self.indexes.sort()
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

    def compute_invocation(self, member: Member):
        inner_invocation = self.compute_structure(member)[0]
        if len(member.get_inner_invocations()) == 0:
            return 0.0
        else:
            return inner_invocation / len(member.get_inner_invocations())


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

    def get_results(self):
        results = []
        self.remove_empty_group()
        for group in self.groups:
            results.append([mem.get_name() for mem in group])
        return results


# cache = True
cache = False


class ClusterAlgorithm(GroupInterface):

    def __init__(self, god_class: GodClass):
        super().__init__(god_class)
        data_path = 'cluster_pickle_data0/' + str(self.god_class.get_last_index())
        try:
            if not cache:
                raise OSError
            with open(data_path, 'rb') as f:
                history_data: ClusterDataContainer = pickle.load(f)
            self.PS = history_data.PS
            self.SimD = history_data.SimD
            self.CDM = history_data.CDM
            self.CS = history_data.CS
            self.SS = history_data.SS
        except OSError:
            self.PS = self.get_PS()
            self.SimD = self.get_SimD()
            self.CDM = self.get_CDM()
            self.CS = self.get_CS()
            self.SS = self.get_SS()
            with open(data_path, 'wb') as f:
                pickle.dump(
                    ClusterDataContainer(self.god_class, self.PS, self.SimD, self.CDM,
                                         self.CS, self.SS), f)
        self.graph = None
        self.edges = None
        self.weights = None
        self.index_chart = self.get_method_index_chart()
        self.thresholds = self.get_top(5, 8)

    def get_PS(self):
        ps = [[0.0 for j in range(len(self.methods))] for i in range(len(self.methods))]
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                ps[i][j] = (1.0 / abs(self.methods[i].get_index() - self.methods[j].get_index()))
                ps[j][i] = (1.0 / abs(self.methods[i].get_index() - self.methods[j].get_index()))
        return ps

    def get_SimD(self):
        simd = [[0.0 for j in range(len(self.methods))] for i in range(len(self.methods))]
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                set1 = set(self.methods[i].get_visits())
                set2 = set(self.methods[j].get_visits())
                union = set1.union(set2)
                intersection = set1.intersection(set2)
                if len(union) > 0:
                    simd[i][j] = len(intersection) / len(union)
                    simd[j][i] = len(intersection) / len(union)
        return simd

    def get_CDM(self):
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

    def get_CS(self):
        embeddings = st5model.encode([method.get_full_text() for method in self.methods])
        similarity = cosine_similarity(embeddings)
        for i in range(len(similarity)):
            for j in range(len(similarity[i])):
                similarity[i][j] = min(1.0, max(0.0, (similarity[i][j] + 0.2) * 0.833))
        return similarity.tolist()

    def get_SS(self):
        texts = [mem.get_code_summary() for mem in self.methods]
        vectors = bgemodel.encode(texts)
        similarity = cosine_similarity(vectors)
        for i in range(len(similarity)):
            for j in range(len(similarity[i])):
                similarity[i][j] = min(1.0, max(0.0, (similarity[i][j] - 0.667) * 3))
        return similarity

    def get_method_index_chart(self):
        chart = [len(self.members)]
        for i in range(len(self.members)):
            if self.members[i] in self.methods:
                chart.append(self.methods.index(self.members[i]))
            else:
                chart.append(len(self.members))
        return chart

    def do_algorithm1(self):
        nodes = self.methods[:]
        edges = []
        weights = []
        n = len(self.methods)
        graph = nx.Graph()
        for i in range(len(nodes)):
            graph.add_node(nodes[i])
        self.graph = graph
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                weight = self.get_weight1(i, j)
                if weight > 0.0:
                    edges.append((self.methods[i], self.methods[j]))
                    weights.append(weight)
        self.edges = edges
        self.weights = weights
        for i in range(len(edges)):
            self.graph.add_edge(edges[i][0], edges[i][1], weight=weights[i])
        self.graph_to_2_chain()
        self.chain_to_groups()

    def get_top(self, m, f):
        MS = []
        FS = []
        # SS = []
        for i in range(len(self.methods)):
            for j in range(i + 1, len(self.methods)):
                MS.append(self.SimD[i][j] + self.CDM[i][j])
                FS.append(self.CS[i][j] + self.SS[i][j])
                # SS.append(self.code_summary_similarity[i_][j])
        MS.sort(reverse=True)
        MS_result = MS[math.ceil(len(MS) * m / 100)]
        FS.sort(reverse=True)
        FS_result = FS[math.ceil(len(FS) * f / 100)]
        return MS_result, FS_result

    def get_weight1(self, index1: int, index2: int):
        weight = 0
        if self.CS[index1][index2] + self.SS[index1][index2] >= self.thresholds[1]:
            weight += 0.5 * (self.CS[index1][index2] + self.SS[index1][index2])
        return weight

    def graph_to_2_chain(self):
        if len(self.weights) == 0:
            return
        threshold = min(self.weights)
        while True:
            if self.get_chain_number() >= 2 or threshold > 2:
                return
            edges = self.edges[:]
            weights = self.weights[:]
            for i in range(len(edges)):
                if self.graph.get_edge_data(edges[i][0], edges[i][1])['weight'] <= threshold:
                    self.edges.remove(edges[i])
                    self.weights.remove(self.graph.get_edge_data(edges[i][0], edges[i][1])['weight'])
                    self.graph.remove_edge(edges[i][0], edges[i][1])
            threshold += 0.02

    def get_chain_number(self):
        chain_number = 0
        for component in nx.connected_components(self.graph):
            if len(component) > 2:
                chain_number += 1
        return chain_number

    def chain_to_groups(self):
        groups = []
        for component in nx.connected_components(self.graph):
            new_group = Group()
            for member in component:
                new_group.append(member)
            groups.append(new_group)
        self.groups = groups

    def do_algorithm2(self):
        self.split_group_by_position()
        self.merge_adjacent_single_method()

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

    def merge_adjacent_single_method(self):
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

    def do_algorithm3(self):
        while len(self.groups) > 2:
            weight_dict = {}
            for i in range(len(self.groups)):
                for j in range(i + 1, len(self.groups)):
                    if i != j:
                        total_weight, num1, num2 = self.get_group_weight(self.groups[i], self.groups[j])
                        weight_dict[total_weight / (len(self.groups[i]) * len(self.groups[j]))] = (i, j)
            merge_pair = self.find_most_similar_groups(weight_dict)
            if merge_pair is None:
                break
            self.groups[merge_pair[0]] = self.groups[merge_pair[0]] + self.groups[merge_pair[1]]
            self.groups.pop(merge_pair[1])

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
        if self.PS[index1][index2] >= 0.3:
            weight += self.PS[index1][index2]
        if self.SimD[index1][index2] + self.CDM[index1][index2] >= self.thresholds[0]:
            weight += (self.SimD[index1][index2] + self.CDM[index1][index2])
        if self.CS[index1][index2] + self.SS[index1][index2] >= self.thresholds[1]:
            weight += 0.5 * (self.CS[index1][index2] + self.SS[index1][index2])
        return weight

    def find_most_similar_groups(self, weight_dict: dict):
        pairs = [v for k, v in sorted(weight_dict.items(), reverse=True)]
        for pair in pairs:
            size_of_new_group = len(self.groups[pair[0]]) + len(self.groups[pair[1]])
            if size_of_new_group < 0.95 * len(self.methods) and len(self.methods) - size_of_new_group > 3:
                return pair
        return None

    def handel_fields(self):
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


class ClusterDataContainer:

    def __init__(self, god_class, PS, SimD, CDM, CS, SS):
        self.god_class = god_class
        self.PS = PS
        self.SimD = SimD
        self.CDM = CDM
        self.CS = CS
        self.SS = SS

    def get_index(self):
        return self.god_class.get_index()
