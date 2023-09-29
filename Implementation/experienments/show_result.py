import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def draw_precision_recall(data: list[tuple[float, float]], title='precision-recall'):
    plt.figure(title)
    precision = 0
    recall = 0
    invalid_data = 0
    for pr in data:
        plt.scatter(pr[0], pr[1], s=10)
        precision += pr[0]
        recall += pr[1]
        if pr[0] == 0.0 and pr[1] == 0.0:
            invalid_data += 1
    precision = precision / (len(data) - invalid_data)
    recall = recall / (len(data) - invalid_data)
    print("precision %.3f recall %.3f" % (precision, recall))
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))


def draw_accuracy(data: list[float], title='accuracy'):
    plt.figure(title)
    plt.hist(data, bins=10)
    plt.xlim((-0.05, 1.05))
    invalid_data = data.count(0.0)
    print("accuracy %.3f" % float(sum(data) / (len(data) - invalid_data)))


def draw_mojofm(data: list[float], title='mojofm'):
    plt.figure(title)
    plt.hist(data, bins=10)
    plt.xlim((-0.05, 1.05))
    plt.ylim((0.0, 30.0))
    # invalid_data = data.count(0.0)
    # print("MoJoFM %.3f" % float(sum(data) / (len(data) - invalid_data)))
    print("%s %.3f" % (title, float(sum(data) / len(data))))


def draw_parameters(raw_data: list[tuple[float, float, float, float]]):
    x = [i[0] for i in raw_data]
    y = [i[1] for i in raw_data]
    z = [i[2] for i in raw_data]
    values = [i[3] for i in raw_data]
    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制散点图
    scatter = ax.scatter(x, y, z, c=values)
    # 添加颜色条
    # plt.imshow(values, cmap='jet')
    # plt.colorbar()
    legend1 = ax.legend(*scatter.legend_elements(), title="Title", loc="upper right", borderaxespad=0, bbox_to_anchor=(1.3, 1))
    ax.add_artist(legend1)
    # 设置坐标轴标签
    ax.set_xlabel('location')
    ax.set_ylabel('structure')
    ax.set_zlabel('text')
    mi = values.index(max(values))
    print("get max MoJoFM = %.3f at (%.1f, %.1f, %.1f)" % (values[mi], x[mi], y[mi], z[mi]))


def draw_compare(data: list[tuple[float, float]], title='compare', c=None):
    plt.figure(title)
    # data.sort(key=lambda a: a[0], reverse=True)  # old
    data.sort(key=lambda a: a[1], reverse=True)
    y1 = [i[0] for i in data]  # as baseline
    y2 = [i[1] for i in data]
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    x = [str(i) for i in range(len(data))]
    colors = ['red' if a - b > 0 else 'blue' for a, b in zip(y1, y2)]

    # # 两列
    # plt.bar(np.arange(len(x)) - 0.2, y2, width=0.4, color='gray', label='前')
    # plt.bar(np.arange(len(x)) + 0.2, y1, width=0.4, color=colors, label='后')

    # # 一列
    # y3 = [a - b for a, b in zip(y1, y2)]
    # plt.bar(x, y2, color='gray', label='baseline')
    # plt.bar(x, y3, color=colors, bottom=y2, label='提升')
    # plt.bar(0, 0, color='red', bottom=0, label='降低')

    plt.plot(x, y2, color='blue', label='ClassSplitter')
    plt.plot(x, y1, color='orange', label='baseline')

    # plt.xticks(np.arange(len(x)), x)
    plt.legend()
    plt.gca().set_xticklabels([])

    fall = 0
    equal = 0
    rise = 0
    huge_rise = 0
    huge_percentage = 0.4
    average0 = 0
    average1 = 0
    for pair in data:
        if pair[0] > pair[1]:
            fall += 1
        elif pair[0] < pair[1]:
            rise += 1
        else:
            equal += 1
        if pair[1] - pair[0] > pair[0] * huge_percentage:
            huge_rise += 1
        average0 += pair[0]
        average1 += pair[1]
    average0 = average0 / len(data)
    average1 = average1 / len(data)
    print("wins: %d, ties: %d, losses: %d, huge rise(%.0f): %d , out of total number %d" % (rise, equal, fall, (huge_percentage * 100), huge_rise, len(data)))
    print("%s MoJoFM: %.3f to %.3f" % (title, average0, average1))

    # Calculate t-test and p-value
    t_statistic, p_value = stats.ttest_ind(y1, y2)
    print("t-statistic:", t_statistic)
    print("p-value:", p_value)
