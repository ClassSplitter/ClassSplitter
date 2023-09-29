import math
import pickle
import random

import matplotlib.pyplot as plt
import numpy.random
from tqdm import tqdm
import statistics
import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
from scipy.stats import kendalltau
import pingouin as pg
import numpy as np

import show_result as sr
from baseline1 import Baseline1
from baseline2 import Baseline2
from baseline3 import Baseline3
from approach import Cluster, ClusterByStep, ClusterSeparate
from god_class import GodClass, DictHandler
from lda import LDAHandler


dict_handler = DictHandler('GodClassRefactorDictionary.csv')

# def get_filtered_classes(dictionary_path):
#     dict_handler = DictHandler(dictionary_path)
#     filtered_classes = []
#     for class_ in dict_handler.get_classes():
#         if class_.get_method_number() > 10 and 0.2 < class_.get_moved_percentage() < 0.7:
#             filtered_classes.append(class_)
#     return filtered_classes


def get_filtered_classes(dictionary_path):
    dict_handler = DictHandler(dictionary_path)
    filtered_classes = []
    for class_ in dict_handler.get_classes():
        if class_.get_checked() is not None and class_.get_checked():
            filtered_classes.append(class_)
    return filtered_classes


def evaluate_lda_result(god_class: GodClass, theta):
    k = len(theta[0])
    scores = [0.0 for i in range(k)]
    for i in range(len(god_class)):
        if god_class.data_frame.at[i, 'type'] == 'Method':
            if god_class.data_frame.at[i, 'moved']:
                for j in range(k):
                    scores[j] += theta[i][j]
            else:
                for j in range(k):
                    scores[j] -= theta[i][j]
    for j in range(k):
        scores[j] = abs(scores[j])
    sc = sum(scores) / god_class.get_method_number()
    bsc = abs(1 - 2 * god_class.get_removed_percentage())
    return max(0.0, (sc - bsc) / (1.0 - bsc))


def show_lda_score():
    
    god_classes = dict_handler.get_filtered_classes()
    lda_scores = []
    for class_ in god_classes:
        lda = LDAHandler(class_, use_gpt_texts=True)
        lda.train(k=math.ceil(class_.get_method_number() / 4.0))
        lda_scores.append(evaluate_lda_result(class_, lda.get_theta()))
    print(lda_scores)
    print('average: %s' % str(sum(lda_scores) / len(lda_scores)))
    plt.figure("score")
    plt.hist(lda_scores, bins=20)
    plt.xlim(0, 1)
    plt.ylim(0, 40)
    plt.show()


def collect_gpt_texts():
    classes = dict_handler.get_filtered_classes()
    for class_ in classes:
        try:
            text_str = str(class_.data_frame.at[0, 'code summary gpt4'])
            print(class_.data_path + " already have text.")
        except KeyError:
            class_.get_gpt_respond()
            class_.to_csv(class_.data_path)


def run_and_get_precision_recall():
    good_classes = dict_handler.get_filtered_classes()
    pr_data_good = []
    pr_data_bad = []
    pr_data_average = []
    ac_data_good = []
    ac_data_bad = []
    ac_data_average = []
    for c in tqdm(good_classes):
        prs = []
        acs = []
        for i in range(4):
            cluster = Cluster(c)
            cluster.group_modify()
            cluster.merge_group()
            prs.append(cluster.get_precision_recall())
            acs.append(cluster.get_accuracy())
        prs.sort(key=lambda x: x[0] + x[1])
        acs.sort()
        pr_data_good.append(prs[-1])
        pr_data_bad.append(prs[0])
        pr_data_average.append((sum([x[0] for x in prs]) / len(prs), sum([x[1] for x in prs]) / len(prs)))
        ac_data_good.append(acs[-1])
        ac_data_bad.append(acs[0])
        ac_data_average.append(sum(acs) / len(acs))
    sr.draw_precision_recall(pr_data_good, title='precision recall good')
    sr.draw_precision_recall(pr_data_bad, title='precision recall bad')
    sr.draw_precision_recall(pr_data_average, title='precision recall average')
    sr.draw_accuracy(ac_data_good, title='accuracy good')
    sr.draw_accuracy(ac_data_bad, title='accuracy bad')
    sr.draw_accuracy(ac_data_average, title='accuracy average')
    plt.show()


def run_and_get_cluster_precision_recall():
    good_classes = dict_handler.get_filtered_classes()
    # pr_data = []
    # ac_data = []
    m_data = []
    for c in tqdm(good_classes):
        cluster = Cluster(c)
        # cluster.group_modify()
        cluster.set_parameters_weight(1.0, 1.0, 1.0, 1.0, 0.0)  # (4.6, 2.9, 3.8, 0.9)
        cluster.graph_to_groups()
        cluster.merge_group()
        # pr_data.append(cluster.get_precision_recall())
        # ac_data.append(cluster.get_accuracy())
        m_data.append(cluster.get_mojofm())
    # sr.draw_precision_recall(pr_data, title='precision recall')
    # sr.draw_accuracy(ac_data, title='accuracy')
    sr.draw_mojofm(m_data, title='MoJoFM')
    plt.show()


def run_and_get_baseline1_precision_recall():
    good_classes = dict_handler.get_filtered_classes()
    # pr_data = []
    # ac_data = []
    m_data = []
    for c in tqdm(good_classes):
        baseline1 = Baseline1(c)
        # baseline1.graph_to_2chain()
        # baseline1.chain_to_groups()
        baseline1.original_graph_to_chain()
        baseline1.original_chain_to_groups()
        # pr_data.append(baseline1.get_precision_recall())
        # ac_data.append(baseline1.get_accuracy())
        m_data.append(baseline1.get_mojofm())
    # sr.draw_precision_recall(pr_data, title='precision recall')
    # sr.draw_accuracy(ac_data, title='accuracy')
    sr.draw_mojofm(m_data, title='MoJoFM')
    plt.show()


def run_and_get_origin_baseline1_precision_recall():
    good_classes = dict_handler.get_filtered_classes()
    # pr_data = []
    # ac_data = []
    m_data = []
    for c in tqdm(good_classes):
        baseline1 = Baseline1(c)
        baseline1.original_graph_to_chain()
        baseline1.original_chain_to_groups()
        # pr_data.append(baseline1.get_precision_recall())
        # ac_data.append(baseline1.get_accuracy())
        m_data.append(baseline1.get_mojofm())
    # sr.draw_precision_recall(pr_data, title='precision recall')
    # sr.draw_accuracy(ac_data, title='accuracy')
    sr.draw_mojofm(m_data, title='MoJoFM')
    plt.show()


def run_and_get_baseline2_precision_recall():
    good_classes = dict_handler.get_filtered_classes()
    # pr_data = []
    # ac_data = []
    m_data = []
    for c in tqdm(good_classes):
        baseline2 = Baseline2(c)
        baseline2.greedy_merge_classes()
        # pr_data.append(baseline2.get_precision_recall())
        # ac_data.append(baseline2.get_accuracy())
        baseline2.get_max_beta_group()
        m_data.append(baseline2.get_mojofm())
    # sr.draw_precision_recall(pr_data, title='precision recall')
    # sr.draw_accuracy(ac_data, title='accuracy')
    sr.draw_mojofm(m_data, title='MoJoFM')
    plt.show()


def run_and_get_baseline3(t=0.36, field=False):
    good_classes = dict_handler.get_filtered_classes()
    m_data = []
    fm_data = []
    for c in tqdm(good_classes):
        baseline3 = Baseline3(c)
        baseline3.original_graph_to_chain(t=t)
        baseline3.original_chain_to_groups()
        if field:
            baseline3.handel_fields()
            fm_data.append(baseline3.get_field_mojofm())
            m_data.append(baseline3.get_mojofm(field=True))
        else:
            m_data.append(baseline3.get_mojofm(field=False))
    sr.draw_mojofm(m_data, title='MoJoFM')
    if field:
        sr.draw_mojofm(fm_data, title='field-MoJoFM')
    plt.show()


def run_and_get_random_precision_recall(show=True):
    good_classes = dict_handler.get_filtered_classes()
    pr_data = []
    ac_data = []
    m_data = []
    for c in tqdm(good_classes):
        baseline_r = BaselineRandom(c)
        # baseline_r.random_allocate()
        baseline_r.random_split()
        pr_data.append(baseline_r.get_precision_recall())
        ac_data.append(baseline_r.get_accuracy())
        m_data.append(baseline_r.get_mojofm())
    if show:
        sr.draw_precision_recall(pr_data, title='precision recall')
        sr.draw_accuracy(ac_data, title='accuracy')
        sr.draw_mojofm(m_data, title='MoJoFM')
        plt.show()
    return statistics.mean(m_data)


def search_parameter():
    result = []
    good_classes = dict_handler.get_filtered_classes()
    clusters = []
    for c in tqdm(good_classes):
        cluster = Cluster(c)
        clusters.append(cluster)
    for i in range(11):
        l = 4.5 + 0.1 * i
        for j in range(11):
            s = 2.5 + 0.1 * j
            for k in range(11):
                t = 3.5 + 0.1 * k
                if i + j + k == 0:
                    continue
                try:
                    print("%d , now: (%.1f, %.1f, %.1f)" % ((100 * i + 10 * j + k), l, s, t))
                    m_data = []
                    for cluster in clusters:
                        cluster.set_parameters_weight(l, s, t)
                        cluster.graph_to_groups()
                        cluster.merge_group()
                        m_data.append(cluster.get_mojofm())
                    mojofm = sum(m_data) / len(m_data)
                    result.append((l, s, t, mojofm))
                except Exception:
                    continue
    with open('pickle_data/parameters_p1.pkl', 'wb') as f:
        pickle.dump(result, f)
    print(result)


def search_gpt_parameter(path):
    result = []
    good_classes = dict_handler.get_filtered_classes()
    clusters = []
    for c in tqdm(good_classes):
        cluster = Cluster(c)
        clusters.append(cluster)
    for i in range(0, 20):
        r = i * 0.1
        try:
            this_weight = (1.0, 1.0, 1.0, r)
            print("now: (%.1f, %.1f, %.1f, %.1f)" % (this_weight[0], this_weight[1], this_weight[2], r))
            m_data = []
            for cluster in clusters:
                cluster.set_parameters_weight(1.0, 1.0, 1.0, r)
                cluster.graph_to_groups()
                cluster.merge_group()
                m_data.append(cluster.get_mojofm())
            invalid_data = m_data.count(0.0)
            mojofm = sum(m_data) / len(m_data)
            result.append((r, mojofm))
        except Exception:
            continue
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    print(result)


def show_parameter_result(file_name):
    with open(file_name, 'rb') as f:
        result: list[tuple] = pickle.load(f)
    sr.draw_parameters(result)
    plt.show()


def show_parameter_gpt_result(file_name):
    with open(file_name, 'rb') as f:
        result: list[tuple] = pickle.load(f)
    x, y = zip(*result)
    plt.plot(x, y)
    plt.show()


def compare_baseline1_cluster():
    good_classes = dict_handler.get_filtered_classes()
    pr_data = []
    ac_data = []
    m_data = []
    for c in tqdm(good_classes):
        cluster = Cluster(c)
        # cluster.group_modify()
        cluster.set_parameters_weight(4.6, 2.9, 3.8, 0.9)  # (4.6, 2.9, 3.8)
        cluster.graph_to_groups()
        cluster.merge_group()
        baseline1 = Baseline1(c)
        # baseline1.graph_to_2chain()
        # baseline1.chain_to_groups()
        baseline1.original_graph_to_chain()
        baseline1.original_chain_to_groups()
        pr_data.append((baseline1.get_precision_recall(), cluster.get_precision_recall()))
        ac_data.append((baseline1.get_accuracy(), cluster.get_accuracy()))
        m_data.append((baseline1.get_mojofm(), cluster.get_mojofm()))
    # sr.draw_precision_recall(pr_data, title='precision recall')
    # sr.draw_accuracy(ac_data, title='accuracy')
    # sr.draw_mojofm(m_data, title='MoJoFM')
    sr.draw_compare(m_data)
    plt.show()


def compare_clusters():
    good_classes = dict_handler.get_filtered_classes()
    pr_data = []
    ac_data = []
    m_data = []
    for c in tqdm(good_classes):
        cluster1 = Cluster(c)
        # cluster.group_modify()
        cluster1.set_parameters_weight(4.6, 2.9, 3.8, 0.9, 0.0)  # (4.6, 2.9, 3.8)
        cluster1.graph_to_groups()
        cluster1.merge_group()
        cluster2 = Cluster(c)  # as baseline
        cluster2.set_parameters_weight(0.0, 2.9, 3.8, 0.9)  # (4.6, 2.9, 3.8)
        cluster2.graph_to_groups()
        cluster2.merge_group()
        pr_data.append((cluster2.get_precision_recall(), cluster1.get_precision_recall()))
        ac_data.append((cluster2.get_accuracy(), cluster1.get_accuracy()))
        m_data.append((cluster2.get_mojofm(), cluster1.get_mojofm()))
    # sr.draw_precision_recall(pr_data, title='precision recall')
    # sr.draw_accuracy(ac_data, title='accuracy')
    # sr.draw_mojofm(m_data, title='MoJoFM'))
    sr.draw_compare(m_data)
    plt.show()


def run_and_get_cluster_middle_state():
    good_classes = dict_handler.get_filtered_classes()
    m_data1 = []
    m_data2 = []
    for c in tqdm(good_classes):
        cluster = Cluster(c)
        # cluster.group_modify()
        cluster.set_parameters_weight(4.6, 2.9, 3.8, 0.9)  # (4.6, 2.9, 3.8)
        cluster.graph_to_groups()
        m_data1.append(cluster.get_mojofm())
        cluster.merge_group()
        m_data2.append(cluster.get_mojofm())
    sr.draw_mojofm(m_data1, title='MoJoFM middle')
    sr.draw_mojofm(m_data2, title='MoJoFM final')
    m_data = [(i, j) for i, j in zip(m_data1, m_data2)]
    sr.draw_compare(m_data)
    plt.show()


def run_and_get_cluster_by_step():
    good_classes = dict_handler.get_filtered_classes()
    m_data = []
    for c in tqdm(good_classes):
        cluster_step = ClusterByStep(c)
        cluster_step.topic_cluster_to_group()
        cluster_step.split_group_by_position()
        cluster_step.merge_split_group(4.6, 2.9, 3.8, 0.9, 0.0)
        m_data.append(cluster_step.get_mojofm())
    sr.draw_mojofm(m_data, title='MoJoFM')
    plt.show()


def compare_steps():
    good_classes = dict_handler.get_filtered_classes()
    m_data1 = []
    m_data2 = []
    for c in tqdm(good_classes):
        cluster_step = ClusterByStep(c)
        cluster_step.topic_cluster_to_group()
        m_data1.append(cluster_step.get_mojofm())
        cluster_step.split_group_by_position()
        m_data2.append(cluster_step.get_mojofm())
    sr.draw_mojofm(m_data1, title='MoJoFM1')
    sr.draw_mojofm(m_data2, title='MoJoFM2')
    m_data = [(i, j) for i, j in zip(m_data1, m_data2)]
    sr.draw_compare(m_data)
    plt.show()


def compare_clusters_with_step():
    good_classes = dict_handler.get_filtered_classes()
    m_data1 = []
    m_data2 = []
    for c in tqdm(good_classes):
        cluster_step = ClusterByStep(c)
        cluster_step.topic_cluster_to_group()
        cluster_step.split_group_by_position()
        cluster_step.merge_split_group(4.6, 2.9, 3.8, 0.9, 0.0)
        m_data2.append(cluster_step.get_mojofm())
    for c in tqdm(good_classes):
        cluster1 = Cluster(c)
        # cluster.group_modify()
        cluster1.set_parameters_weight(4.6, 2.9, 3.8, 0.9, 0.0)  # (4.6, 2.9, 3.8)
        cluster1.graph_to_groups()
        cluster1.merge_group()
        m_data1.append(cluster1.get_mojofm())
    sr.draw_mojofm(m_data1, title='MoJoFM1')
    sr.draw_mojofm(m_data2, title='MoJoFM2')
    m_data = [(i, j) for i, j in zip(m_data1, m_data2)]
    sr.draw_compare(m_data)
    plt.show()


def run_step_clusters(data_path):
    mojofm_data = []
    try:
        with open(data_path, 'rb') as f:
            history_data: list[float] = pickle.load(f)
        mojofm_data += history_data
    except OSError:
        with open(data_path, 'wb') as f:
            pickle.dump(mojofm_data, f)
    good_classes = dict_handler.get_filtered_classes()
    for i in range(100):
        m_data = []
        for c in tqdm(good_classes):
            cluster_step = ClusterByStep(c)
            cluster_step.topic_cluster_to_group()
            cluster_step.split_group_by_position()
            cluster_step.merge_split_group(4.6, 2.9, 3.8, 0.9, 0.0)
            m_data.append(cluster_step.get_mojofm())
        mojofm_data.append(sum(m_data) / len(m_data))
        print("%.3f" % (sum(m_data) / len(m_data)))
        with open(data_path, 'wb')as f:
            pickle.dump(mojofm_data, f)


def show_step_clusters(data_path):
    try:
        with open(data_path, 'rb') as f:
            history_data: list[float] = pickle.load(f)
        print(history_data)
        print("average: %.3f" % statistics.mean(history_data))
        print("variance: %.3f" % statistics.pstdev(history_data))
    except OSError:
        print("file not found.")


def run_and_get_gpt_method():
    good_classes = dict_handler.get_filtered_classes()
    m_data = []
    for c in tqdm(good_classes):
        gpt_method = GPTRefactor(c)
        m_data.append(gpt_method.get_mojofm())
    sr.draw_mojofm(m_data, title='MoJoFM')
    plt.show()


def create_clusters():
    good_classes = dict_handler.get_filtered_classes()
    for c in tqdm(good_classes):
        cluster = Cluster(c)


def code_summary_effect():
    good_classes = dict_handler.get_filtered_classes()
    clusters = []
    for c in tqdm(good_classes):
        clusters.append(Cluster(c))
    random.seed(1)
    similarities_between_same_class = []
    similarities_between_different_class = []
    for c in tqdm(clusters):
        method_num = len(c.methods)
        for times in range(200):
            m1, m2 = random.sample(range(0, method_num), 2)
            sim = c.SS[m1][m2]
            sim = 0.0 * c.PS[m1][m2] + 0.0 * c.SimD[m1][m2] + 0.0 * c.CDM[m1][m2] + 0.5 * c.CS[m1][m2] + 0.5 * c.SS[m1][m2]
            if c.methods[m1].is_removed() == c.methods[m2].is_removed():
                similarities_between_same_class.append(sim)
            else:
                similarities_between_different_class.append(sim)
    print("same average: %.3f" % (sum(similarities_between_same_class) / len(similarities_between_same_class)))
    print("different average: %.3f" % (sum(similarities_between_different_class) / len(similarities_between_different_class)))
    range_bottom = min(min(similarities_between_same_class), min(similarities_between_different_class))
    range_ceil = max(max(similarities_between_same_class), max(similarities_between_different_class))
    step = 20
    x = []
    larger = []
    lower = []
    larger_same_p = []
    lower_same_p = []
    for t in range(0, step):
        threshold = range_bottom + (t / step) * (range_ceil - range_bottom)
        same_larger = sum(1 for s in similarities_between_same_class if s >= threshold)
        diff_larger = sum(1 for s in similarities_between_different_class if s >= threshold)
        same_lower = len(similarities_between_same_class) - same_larger
        diff_lower = len(similarities_between_different_class) - diff_larger
        x.append(threshold)
        larger.append(same_larger + diff_larger)
        lower.append(same_lower + diff_lower)
        larger_same_p.append(1.0 if (same_larger + diff_larger) == 0 else same_larger / (same_larger + diff_larger))
        lower_same_p.append(0.0 if (same_lower + diff_lower) == 0 else same_lower / (same_lower + diff_lower))
    fig = plt.figure(figsize=(8, 6))
    # plt.grid(axis="y", linestyle='-.')
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0, larger[0] + lower[0]])
    # ax1.bar(x, lower, color='gray', bottom=0, label='lower')
    # ax1.bar(x, larger, color='pink', bottom=lower, label='larger')
    ax1.plot(x, lower, label='lower_percentage', color='red', ms=10, mfc='red', lw=3, marker=None)
    ax1.fill_between(x, lower, [(larger[0] + lower[0]) for i in lower], where=None, color='red', alpha=0.1)
    # ax1.set_xticks([])
    ax1.set_xlim([range_bottom - (1 / step) * (range_ceil - range_bottom), range_ceil])
    ax2 = ax1.twinx()
    ax2.set_ylim([0.0, 1.0])
    ax2.plot(x, larger_same_p, label='larger_percentage', color='blue', ms=10, mfc='yellow', lw=3, marker='o')
    ax2.plot(x, lower_same_p, label='lower_percentage', color='gray', ms=10, mfc='green', lw=3, marker='o')
    ax2.set_xlim([range_bottom - (1 / step) * (range_ceil - range_bottom), range_ceil])
    # ax2.set_xticks([])
    plt.show()


def run_and_get_cluster_separate(field=False):
    good_classes = dict_handler.get_filtered_classes()
    m_data = []
    mm_data = []
    fm_data = []
    for c in tqdm(good_classes):
        cluster_separate = ClusterSeparate(c)
        cluster_separate.separate_to_group()
        cluster_separate.split_group_by_position()
        cluster_separate.merge_adjacent_single_method()
        cluster_separate.merge_separate_group()
        cluster_separate.handel_fields()
        m_data.append(cluster_separate.get_mojofm(field=field))
        mm_data.append(cluster_separate.get_mojofm(field=False))
        if field:
            fm_data.append(cluster_separate.get_field_mojofm())
        # if m_data[-1] > 0.9:
        #     tqdm.write("%d found mojofm %.2f" % (len(m_data) - 1, m_data[-1]))
        # if m_data[-1] <= 0.05:
        #     tqdm.write("%s found mojofm %.2f" % (len(m_data) - 1, m_data[-1]))
    if field:
        sr.draw_mojofm(fm_data, title='field-MoJoFM')
    sr.draw_mojofm(m_data, title='MoJoFM')
    sr.draw_mojofm(mm_data, title='method-MoJoFM')
    plt.show()


def compare_clusters_with_separate():
    good_classes = dict_handler.get_filtered_classes()
    m_data1 = []
    m_data2 = []
    for c in tqdm(good_classes):
        cluster_separate = ClusterSeparate(c)
        cluster_separate = ClusterSeparate(c)
        cluster_separate.separate_to_group()
        cluster_separate.split_group_by_position()
        cluster_separate.merge_adjacent_single_method()
        # cluster_separate.merge_split_group(1.0, 1.0, 1.0, 1.0, 0.0)
        cluster_separate.merge_separate_group()
        m_data2.append(cluster_separate.get_mojofm())
    for c in tqdm(good_classes):
        cluster1 = Cluster(c)
        # cluster.group_modify()
        cluster1.set_parameters_weight(1.0, 1.0, 1.0, 1.0, 0.0)  # (4.6, 2.9, 3.8)
        cluster1.graph_to_groups()
        cluster1.merge_group()
        m_data1.append(cluster1.get_mojofm())
    sr.draw_mojofm(m_data1, title='MoJoFM1')
    sr.draw_mojofm(m_data2, title='MoJoFM2')
    m_data = [(i, j) for i, j in zip(m_data1, m_data2)]
    sr.draw_compare(m_data)
    plt.show()


def compare_separates(field):
    good_classes = dict_handler.get_filtered_classes()
    m_data1 = []
    m_data2 = []
    fm_data1 = []
    fm_data2 = []
    for c in tqdm(good_classes):
        cluster_separate = ClusterSeparate(c)
        cluster_separate.set_use_similarity([1, 1, 1, 1])
        cluster_separate.set_use_back_algorithm(False)
        cluster_separate.separate_to_group()
        cluster_separate.split_group_by_position()
        cluster_separate.merge_adjacent_single_method()
        cluster_separate.merge_separate_group()
        cluster_separate.handel_fields()
        m_data1.append(cluster_separate.get_mojofm(field=field))
        if field:
            fm_data1.append(cluster_separate.get_field_mojofm())
    for c in tqdm(good_classes):
        cluster_separate = ClusterSeparate(c)
        cluster_separate.separate_to_group()
        cluster_separate.split_group_by_position()
        cluster_separate.merge_adjacent_single_method()
        cluster_separate.merge_separate_group()
        cluster_separate.handel_fields()
        m_data2.append(cluster_separate.get_mojofm(field=field))
        if field:
            fm_data2.append(cluster_separate.get_field_mojofm())
    if field:
        m1 = 0
        m2 = 0
        fm1 = 0
        fm2 = 0
        fn = 0
        en = 0
        for i in range(len(good_classes)):
            field_number = good_classes[i].get_field_number()
            fn += field_number
            entity_number = good_classes[i].get_field_number() + good_classes[i].get_method_number()
            en += entity_number
            m1 += m_data1[i]
            m2 += m_data2[i]
            fm1 += fm_data1[i]
            fm2 += fm_data2[i]
        f_average = (fm1 / len(good_classes), fm2 / len(good_classes))
        e_average = (m1 / len(good_classes), m2 / len(good_classes))
        print("MoJoFM of field: %.3f , %.3f" % f_average)
        print("MoJoFM of entity: %.3f , %.3f" % e_average)
    else:
        m1 = 0
        m2 = 0
        mn = 0
        for i in range(len(good_classes)):
            method_number = good_classes[i].get_method_number()
            mn += method_number
            m1 += m_data1[i]
            m2 += m_data2[i]
        m_average = (m1 / len(good_classes), m2 / len(good_classes))
        print("MoJoFM of method: %.3f , %.3f" % m_average)
    # sr.draw_mojofm(m_data1, title='MoJoFM1')
    # sr.draw_mojofm(m_data2, title='MoJoFM2')
    m_data = [(i, j) for i, j in zip(m_data1, m_data2)]
    sr.draw_compare(m_data)
    plt.show()


def compare_baseline1_separates(field=False):
    good_classes = dict_handler.get_filtered_classes()
    m_data1 = []
    m_data2 = []
    fm_data1 = []
    fm_data2 = []
    mm_data1 = []
    mm_data2 = []
    for c in tqdm(good_classes):
        baseline1 = Baseline1(c)
        baseline1.original_graph_to_chain()
        baseline1.original_chain_to_groups()
        baseline1.handel_fields()
        # if len(baseline1.groups) == 1:
        #     continue
        m_data1.append((baseline1.get_mojofm(field=field)))
        mm_data1.append(baseline1.get_mojofm(field=False))
        if field:
            fm_data1.append((baseline1.get_field_mojofm()))
            # if fm_data1[-1] >= 0.95 and fm_data2[-1] <= 0.05:
            #     tqdm.write("%s found field mojofm %.2f %.2f" % (len(m_data1) - 1, fm_data1[-1], fm_data2[-1]))
        cluster_separate = ClusterSeparate(c)
        cluster_separate.set_use_similarity([1, 1, 1, 1])
        cluster_separate.separate_to_group()
        cluster_separate.split_group_by_position()
        cluster_separate.merge_adjacent_single_method()
        cluster_separate.merge_separate_group()
        cluster_separate.handel_fields()
        m_data2.append(cluster_separate.get_mojofm(field=field))
        mm_data2.append(cluster_separate.get_mojofm(field=False))
        if field:
            fm_data2.append(cluster_separate.get_field_mojofm())
    # if field:
    #     m1 = 0
    #     m2 = 0
    #     fm1 = 0
    #     fm2 = 0
    #     fn = 0
    #     en = 0
    #     for i in range(len(good_classes)):
    #         field_number = good_classes[i].get_field_number()
    #         fn += field_number
    #         entity_number = good_classes[i].get_field_number() + good_classes[i].get_method_number()
    #         en += entity_number
    #         m1 += m_data1[i]
    #         m2 += m_data2[i]
    #         fm1 += fm_data1[i]
    #         fm2 += fm_data2[i]
    #     f_average = (fm1 / len(good_classes), fm2 / len(good_classes))
    #     e_average = (m1 / len(good_classes), m2 / len(good_classes))
    #     print("MoJoFM of field: %.3f , %.3f" % f_average)
    #     print("MoJoFM of entity: %.3f , %.3f" % e_average)
    # else:
    #     m1 = 0
    #     m2 = 0
    #     mn = 0
    #     for i in range(len(good_classes)):
    #         method_number = good_classes[i].get_method_number()
    #         mn += method_number
    #         m1 += method_number * m_data1[i]
    #         m2 += method_number * m_data2[i]
    #     m_average = (m1 / mn, m2 / mn)
    #     print("MoJoFM of method: %.3f , %.3f" % m_average)
    # sr.draw_mojofm(m_data1, title='MoJoFM1')
    # sr.draw_mojofm(m_data2, title='MoJoFM2')
    m_data = [(i, j) for i, j in zip(m_data1, m_data2)]
    mm_data = [(i, j) for i, j in zip(mm_data1, mm_data2)]
    sr.draw_compare(m_data)
    sr.draw_compare(mm_data, title='method-compare')
    if field:
        fm_data = [(i, j) for i, j in zip(fm_data1, fm_data2)]
        sr.draw_compare(fm_data, title='field-compare')
    plt.show()


def get_adjacent_statistics():
    good_classes = dict_handler.get_filtered_classes()
    f_pair_same = []
    f_pair_all = []
    m_pair_same = []
    m_pair_all = []
    m_random_pair_same = []
    f_random_pair_same = []
    m_size = []
    f_size = []
    f_block_num = []
    m_block_num = []
    f_block_sizes = []
    m_block_sizes = []
    f_1_size_block = []
    m_1_size_block = []
    new_class_f_block_size = []
    new_class_m_block_size = []
    new_class_f_block_num = []
    new_class_m_block_num = []
    total_m_pair_number = []
    total_f_pair_number = []
    for c in tqdm(good_classes):
        cluster = Cluster(c)
        field_same = 0
        field_all = 0
        method_same = 0
        method_all = 0
        method_block_num = 0
        method_block_sizes = []
        field_block_num = 0
        field_block_sizes = []
        field_block_size = 1
        method_block_size = 1
        new_class_field_block_sizes = []
        new_class_method_block_sizes = []
        for i in range(len(cluster.members) - 1):
            if cluster.members[i].is_field() and cluster.members[i + 1].is_field():
                field_all += 1
                if cluster.members[i].is_removed() == cluster.members[i + 1].is_removed():
                    field_same += 1
                    field_block_size += 1
                else:
                    field_block_sizes.append(field_block_size)
                    if cluster.members[i].is_removed():
                        new_class_field_block_sizes.append(field_block_size)
                    field_block_size = 1
            if cluster.members[i].is_field() and not cluster.members[i + 1].is_field():
                field_block_sizes.append(field_block_size)
                if cluster.members[i].is_removed():
                    new_class_field_block_sizes.append(field_block_size)
                field_block_size = 0
                if cluster.members[i + 1].is_method():
                    method_block_size = 1
                else:
                    method_block_size = 0
            if cluster.members[i].is_method() and cluster.members[i + 1].is_method():
                method_all += 1
                if cluster.members[i].is_removed() == cluster.members[i + 1].is_removed():
                    method_same += 1
                    method_block_size += 1
                else:
                    method_block_sizes.append(method_block_size)
                    if cluster.members[i].is_removed():
                        new_class_method_block_sizes.append(method_block_size)
                    method_block_size = 1
            if cluster.members[i].is_method() and not cluster.members[i + 1].is_method():
                method_block_sizes.append(method_block_size)
                if cluster.members[i].is_removed():
                    new_class_method_block_sizes.append(method_block_size)
                method_block_size = 0
                if cluster.members[i + 1].is_field():
                    field_block_size = 1
                else:
                    field_block_size = 0
            if not cluster.members[i].is_method() and not cluster.members[i].is_field():
                if cluster.members[i + 1].is_method():
                    method_block_size = 1
                if cluster.members[i + 1].is_field():
                    field_block_size = 1
            if i == len(cluster.members) - 2:
                if method_block_size > 0:
                    method_block_num += 1
                    method_block_sizes.append(method_block_size)
                    if cluster.members[i + 1].is_removed():
                        new_class_method_block_sizes.append(method_block_size)
                if field_block_size > 0:
                    field_block_num += 1
                    field_block_sizes.append(field_block_size)
                    if cluster.members[i + 1].is_removed():
                        new_class_field_block_sizes.append(field_block_size)
        total_m_pair_number.append(c.get_method_number() * (c.get_method_number() - 1) / 2)
        total_f_pair_number.append(c.get_field_number() * (c.get_field_number() - 1) / 2)
        f_pair_same.append(field_same)
        f_pair_all.append(field_all)
        m_pair_same.append(method_same)
        m_pair_all.append(method_all)
        pm = c.get_removed_percentage()
        m_random_pair_same.append((1.0 - pm) ** 2 + pm ** 2)
        pf = c.get_removed_field_percentage()
        f_random_pair_same.append((1.0 - pf) ** 2 + pf ** 2)
        m_size.append(c.get_method_number())
        f_size.append(c.get_field_number())
        average_field_block_size = 0 if len(field_block_sizes) == 0 else sum(field_block_sizes) / len(field_block_sizes)
        average_method_block_size = 0 if len(method_block_sizes) == 0 else sum(method_block_sizes) / len(method_block_sizes)
        f_block_sizes.append(average_field_block_size)
        m_block_sizes.append(average_method_block_size)
        f_block_num.append(len(field_block_sizes))
        m_block_num.append(len(method_block_sizes))
        f_1_size_block.append(field_block_sizes.count(1))
        m_1_size_block.append(method_block_sizes.count(1))
        new_class_f_block_size += new_class_field_block_sizes
        new_class_m_block_size += new_class_method_block_sizes
        new_class_f_block_num.append(len(new_class_field_block_sizes))
        new_class_m_block_num.append(len(new_class_method_block_sizes))
    print("total %d method pairs and %d field pairs" % (sum(total_m_pair_number), sum(total_f_pair_number)))
    print("total %d adjacent field pairs, %d same class, %.1f %%" % (
        sum(f_pair_all), sum(f_pair_same), 100 * sum(f_pair_same) / sum(f_pair_all)))
    print("total %d adjacent method pairs, %d same class, %.1f %%" % (
        sum(m_pair_all), sum(m_pair_same), 100 * sum(m_pair_same) / sum(m_pair_all)))
    average_percentage = sum([i / j for i, j in zip(m_pair_same, m_pair_all)]) / len(m_pair_all)
    print("average %.1f %% of adjacent method pairs go into same class in each class" % (100 * average_percentage))
    random_method_percentage = sum([i * j for i, j in zip(m_random_pair_same, m_size)]) / sum(m_size)
    random_field_percentage = sum([i * j for i, j in zip(f_random_pair_same, f_size)]) / sum(f_size)
    print("random get %.1f %% of adjacent method pairs, %.1f %% of adjacent field pairs in average" % (100 * random_method_percentage, 100 * random_field_percentage))
    ambs = sum([i * j for i, j in zip(m_block_sizes, m_block_num)]) / sum(m_block_num)
    print("average %.2f blocks of methods in each class, average method block size: %.2f" % (ambs, sum(m_block_num) / len(m_block_num)))
    afbs = sum([i * j for i, j in zip(f_block_sizes, f_block_num)]) / sum(f_block_num)
    print("average %.2f blocks of fields in each class, average field block size: %.2f" % (
    afbs, sum(f_block_num) / len(f_block_num)))
    print("average %.2f 1 length method block and %.2f 1 length field block" % (sum(f_1_size_block) / len(f_1_size_block), sum(m_1_size_block) / len(m_1_size_block)))
    print("new class has average %.2f methods and %.2f fields" % (sum(new_class_m_block_size) / len(new_class_m_block_num), sum(new_class_f_block_size) / len(new_class_f_block_num)))
    print("new class has average %.2f method blocks and %.2f field blocks" % (sum(new_class_m_block_num) / len(new_class_m_block_num), sum(new_class_f_block_num) / len(new_class_f_block_num)))
    print("new class has average %.2f isolated method blocks and %.2f isolated field blocks" % (new_class_m_block_size.count(1) / len(new_class_m_block_num), new_class_f_block_size.count(1) / len(new_class_m_block_num)))
    # plt.figure('probabilities')
    # y = [i / j for i, j in zip(m_pair_same, m_pair_all)]
    # y.sort(reverse=True)
    # x = [str(i) for i in range(len(y))]
    # plt.plot(x, y, color='blue', label='proportion of pairs in same class')
    # plt.legend()
    # plt.gca().set_xticklabels([])
    # plt.ylim((0.0, 1.05))
    # sm.graphics.beanplot([new_class_f_block_size, new_class_m_block_size], labels=['field blocks', 'method blocks'], jitter=True)
    plt.figure('boxplot')
    plt.boxplot([new_class_f_block_size, new_class_m_block_size], labels=['field blocks', 'method blocks'], showfliers=False, showmeans=False)
    plt.show()


def get_cluster_separate_time(field=False):
    good_classes = dict_handler.get_filtered_classes()
    time_data = []
    embedding_time_data = []
    cosine_time_data = []
    cluster_time_data = []
    for c in tqdm(good_classes):
        cluster_separate = ClusterSeparate(c)
        cluster_separate.timer.restart()
        # cluster_separate.timer.restart()
        cluster_separate.separate_to_group()
        cluster_separate.split_group_by_position()
        # cluster_separate.merge_adjacent_single_method()
        # cluster_separate.merge_split_group(1.0, 1.0, 1.0, 1.0, 0.0)
        cluster_separate.merge_separate_group()
        cluster_separate.timer.record_past('clustering')
        cluster_separate.handel_fields()
        cluster_separate.timer.record_past('field')
        time_data.append(cluster_separate.timer.get_total_time())
        embedding_time_data.append(cluster_separate.timer.get_time('fullcode encode') + cluster_separate.timer.get_time('summary encode'))
        cosine_time_data.append(cluster_separate.timer.get_time('fullcode sim calculation') + cluster_separate.timer.get_time('summary sim calculation'))
        cluster_time_data.append(cluster_separate.timer.get_time('clustering') + cluster_separate.timer.get_time('field'))
        # if m_data[-1] > 0.9:
        #     tqdm.write("%d found mojofm %.2f" % (len(m_data) - 1, m_data[-1]))
        # if m_data[-1] <= 0.05:
        #     tqdm.write("%s found mojofm %.2f" % (len(m_data) - 1, m_data[-1]))
    print('max time: %.2f' % max(time_data))
    print('min time: %.2f' % min(time_data))
    print('average time: %.2f' % (sum(time_data) / len(time_data)))
    print('average embedding time: %.2f' % (sum(embedding_time_data) / len(embedding_time_data)))
    print('average cosine time: %.2f' % (sum(cosine_time_data) / len(cosine_time_data)))
    print('average clustering time: %.2f' % (sum(cluster_time_data) / len(cluster_time_data)))

def get_baseline1_time():
    good_classes = dict_handler.get_filtered_classes()
    time_data = []
    for c in tqdm(good_classes):
        baseline1 = Baseline1(c)
        baseline1.original_graph_to_chain()
        baseline1.original_chain_to_groups()
        baseline1.handel_fields()
        baseline1.timer.record_past('end')
        time_data.append(baseline1.timer.get_total_time())
    print('max time: %.2f' % max(time_data))
    print('min time: %.2f' % min(time_data))
    print('average time: %.2f' % (sum(time_data) / len(time_data)))


def get_baseline2_time():
    good_classes = dict_handler.get_filtered_classes()
    time_data = []
    for c in tqdm(good_classes):
        baseline2 = Baseline2(c)
        baseline2.greedy_merge_classes()
        baseline2.get_max_beta_group()
        baseline2.timer.record_past('end')
        time_data.append(baseline2.timer.get_total_time())
    print('max time: %.2f' % max(time_data))
    print('min time: %.2f' % min(time_data))
    print('average time: %.2f' % (sum(time_data) / len(time_data)))


def get_baseline3_time():
    good_classes = dict_handler.get_filtered_classes()
    time_data = []
    for c in tqdm(good_classes):
        baseline3 = Baseline3(c)
        baseline3.original_graph_to_chain(t=0.36)
        baseline3.original_chain_to_groups()
        baseline3.handel_fields()
        baseline3.timer.record_past('end')
        time_data.append(baseline3.timer.get_total_time())
    print('max time: %.2f' % max(time_data))
    print('min time: %.2f' % min(time_data))
    print('average time: %.2f' % (sum(time_data) / len(time_data)))



def search_baseline3_parameter():
    good_classes = dict_handler.get_filtered_classes()
    test_cases = [good_classes[10], good_classes[30], good_classes[50], good_classes[70], good_classes[90]]
    result = []
    for i in tqdm(range(41)):
        p = 0.3 + 0.01 * i
        m_data = []
        for c in test_cases:
            baseline3 = Baseline3(c)
            baseline3.original_graph_to_chain(t=p)
            baseline3.original_chain_to_groups()
            m_data.append(baseline3.get_mojofm())
        result.append((p, sum(m_data) / len(m_data)))
    x = [n[0] for n in result]
    y = [n[1] for n in result]
    plt.plot(x, y)
    for r in result:
        print("(%.2f, %.2f)" % r)


def get_human_score():
    dfv_cty = pd.read_excel('F:\workspace\data\GodClass evaluation\questionnaire-final.xlsx', sheet_name='chentianyi',
                           usecols='K:L').values.tolist()[:-2]
    dfv_lb = pd.read_excel('F:\workspace\data\GodClass evaluation\questionnaire-final.xlsx', sheet_name='liubo',
                           usecols='K:L').values.tolist()[:-2]
    dfv_dch = pd.read_excel('F:\workspace\data\GodClass evaluation\questionnaire-final.xlsx', sheet_name='dongchunhao',
                           usecols='K:L').values.tolist()[:-2]
    data = [[int(p[0]) for p in dfv_cty] + [int(p[1]) for p in dfv_cty],
            [int(p[0]) for p in dfv_lb] + [int(p[1]) for p in dfv_lb],
            [int(p[0]) for p in dfv_dch] + [int(p[1]) for p in dfv_dch]]
    kappas = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            kappas.append(cohen_kappa_score(data[i], data[j]))
    print(kappas)
    # df = pd.DataFrame(data)
    # kappa = fleiss_kappa(data)
    # icc = pg.intraclass_corr(data=df)
    # print(icc)


# def run_and_get_cluster_algorithm():
#     good_classes = dict_handler.get_filtered_classes()
#     m_data = []
#     for c in tqdm(good_classes):
#         cluster_algo = ClusterAlgorithm(c)
#         cluster_algo.do_algorithm1()
#         cluster_algo.do_algorithm2()
#         cluster_algo.do_algorithm3()
#         cluster_algo.handel_fields()
#         m_data.append(cluster_algo.get_mojofm())
#     sr.draw_mojofm(m_data, title='MoJoFM')
#     plt.show()



# collect_gpt_texts()

# show_lda_score()

# run_and_get_precision_recall()

# run_and_get_cluster_precision_recall()

# run_and_get_baseline1_precision_recall()

# run_and_get_origin_baseline1_precision_recall()

# run_and_get_baseline2_precision_recall()

# run_and_get_baseline3(t=0.36, field=True)

# run_and_get_random_precision_recall()

# mojofms = []
# for i in range(20):
#     mojofms.append(run_and_get_random_precision_recall(show=False))
# print("average: %.3f" % statistics.mean(mojofms))
# print("variance: %.3f" % statistics.pstdev(mojofms))


# search_parameter()
# show_parameter_result('pickle_data/parameters_p1.pkl')

# 'pickle_data/parameters_gpt_1.pkl' for first cs + w2v, 'pickle_data/parameters_gpt_2.pkl' for first cs + st, 'pickle_data/parameters_gpt_3.pkl' for first cs + st without st5
# 'pickle_data/parameters_gpt_6.pkl' for 4.6, 2.9, 0.0, r in 3* cs + st, 'pickle_data/parameters_gpt_7.pkl' for 1.5, 1.5, 0.0, r in 3* cs + st
# search_gpt_parameter('pickle_data/parameters_gpt_8.pkl')
# show_parameter_gpt_result('pickle_data/parameters_gpt_8.pkl')

# compare_baseline1_cluster()

# compare_clusters()

# run_and_get_cluster_middle_state()

# numpy.random.seed(30000)  # 200 30 30000
# run_and_get_cluster_by_step()

# compare_steps()

# numpy.random.seed(30000)
# compare_clusters_with_step()

# run_step_clusters('pickle_data/step_data_gpt.plk')
# show_step_clusters('pickle_data/step_data_gpt.plk')

# run_and_get_gpt_method()

# create_clusters()

# code_summary_effect()

# run_and_get_cluster_separate(field=True)

# compare_clusters_with_separate()

# compare_separates(field=False)

# compare_baseline1_separates(field=True)

# get_adjacent_statistics()

# get_cluster_separate_time(field=True)

# get_baseline1_time()

# get_baseline2_time()

# get_baseline3_time()

# get_human_score()

# search_baseline3_parameter()

# run_and_get_cluster_algorithm()
