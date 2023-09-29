from approach import Member, Group, GroupInterface
from god_class import GodClass
import pandas as pd
import matplotlib.pyplot as plt
from timer import Timer

class Group2(Group):

    def __init__(self, member: Member):
        super().__init__()
        self.append(member)
        self.cohesion = 0.0

    def update_cohesion(self, cohesion):
        self.cohesion = cohesion


class Baseline2(GroupInterface):
    # 2022

    def __init__(self, god_class: GodClass):
        super().__init__(god_class)
        self.timer = Timer()
        self.SSMs = self.get_SSMs()
        self.groups = self.get_init_groups()
        self.history = []

    def get_member_index(self, member: Member):
        if member in self.members:
            return self.members.index(member)
        else:
            raise IndexError

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

    def get_init_groups(self):
        groups = []
        for i in range(len(self.methods)):
            groups.append(Group2(self.methods[i]))
        return groups

    def get_dependency(self, method1: Member, method2: Member):
        return self.SSMs[self.get_method_index(method1)][self.get_method_index(method2)]

    def get_cohesion(self, group: Group2):
        members = group.members
        n = len(members)
        dependencies = []
        if n <= 1:
            group.update_cohesion(0.0)
            return 0.0
        for i in range(n):
            method1 = members[i]
            for j in range(i + 1, n):
                method2 = members[j]
                dependencies.append(self.get_dependency(method1, method2))
        cohesion = 2 * sum(dependencies) / (n * n - n)
        group.update_cohesion(cohesion)
        return cohesion

    def get_coupling(self, group1: Group2, group2: Group2):
        methods1 = group1.members
        methods2 = group2.members
        dependencies = []
        for i in range(len(methods1)):
            method1 = methods1[i]
            for j in range(len(methods2)):
                method2 = methods2[j]
                dependencies.append(self.get_dependency(method1, method2))
        coupling = sum(dependencies) / (len(methods1) * len(methods2))
        return coupling

    def get_beta(self, groups: list[Group2]):
        p = len(groups)
        if p == 1:
            return self.get_cohesion(groups[0])
        cohesions = []
        for i in range(p):
            cohesions.append(groups[i].cohesion)
        couplings = []
        for i in range(p):
            group1 = groups[i]
            for j in range(i + 1, p):
                group2 = groups[j]
                couplings.append(self.get_coupling(group1, group2))
        beta = sum(cohesions) / p - 2 * sum(couplings) / (p * p - p)
        return beta

    def greedy_merge_classes(self):
        self.history.append(self.groups[:])
        while(len(self.groups)) > 2:
            coupling_dict = {}
            for i in range(len(self.groups)):
                group1 = self.groups[i]
                for j in range(i + 1, len(self.groups)):
                    group2 = self.groups[j]
                    coupling = self.get_coupling(group1, group2)
                    coupling_dict[(group1, group2)] = coupling
            max_coupling_groups = max(coupling_dict.items(), key=lambda x: x[1])[0]
            group1 = self.groups.pop(self.groups.index(max_coupling_groups[0]))
            group2 = self.groups.pop(self.groups.index(max_coupling_groups[1]))
            new_group = group1 + group2
            new_group.update_cohesion(self.get_cohesion(new_group))
            self.groups.append(new_group)
            self.remove_empty_group()
            self.history.append(self.groups[:])

    def get_max_beta_group(self):
        beta_dict = {}
        for i in range(len(self.history)):
            beta = self.get_beta(self.history[i])
            beta_dict[i] = beta
        max_beta_index = max(beta_dict.items(), key=lambda x: x[1])[0]
        self.groups = self.history[max_beta_index]


