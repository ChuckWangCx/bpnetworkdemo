import BPNetDemo
import matplotlib.pyplot as plt
from pylab import *

# 学习率不同，隐藏节点200个，迭代次数5次为例
# [0.9736, 0.9746, 0.9723, 0.9665, 0.9669, 0.9596, 0.9293, 0.878]
learning_rate_array = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7]
learning_rate_array_succeed = [0.9736, 0.9746, 0.9723, 0.9665, 0.9669, 0.9596, 0.9293, 0.878]
# for i in learning_rate_array:
#     learning_rate_array_succeed.append(BPNetDemo.performance(200, i, 5))
# print(learning_rate_array_succeed)

plt.plot(learning_rate_array, learning_rate_array_succeed, marker='o', mec='r', mfc='w',label=u'learning_rate_relation')
plt.legend()  # 让图例生效
plt.xticks(learning_rate_array, learning_rate_array, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"learning_rate") #X轴标签
plt.ylabel("succeed_rate") #Y轴标签
plt.title("A learning_rate plot") #标题

plt.show()

# 迭代次数不同，0.1学习率情况，隐藏节点200个为例
# [0.97, 0.9728, 0.9751, 0.9736, 0.973, 0.9743, 0.9696, 0.9704]
epochs_array = [2, 4, 6, 8, 10, 14, 18, 24]
epochs_array_succeed = [0.97, 0.9728, 0.9751, 0.9736, 0.973, 0.9743, 0.9696, 0.9704]
# for i in epochs_array:
#     epochs_array_succeed.append(BPNetDemo.performance(200, 0.1, i))
# print(epochs_array_succeed)
plt.plot(epochs_array, epochs_array_succeed, marker='o', mec='r', mfc='w',label=u'epochs_relation')
plt.legend()  # 让图例生效
plt.xticks(epochs_array, epochs_array, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"epochs") #X轴标签
plt.ylabel("succeed_rate") #Y轴标签
plt.title("A epochs plot") #标题

plt.show()

# 隐藏层节点不同 学习率为0.5，迭代次数5次为例
# [0.9675, 0.9727, 0.976, 0.9744, 0.9747, 0.9748]
hidden_nodes_array = [100, 200, 400, 600, 1000, 2000]
hidden_nodes_array_succeed = [0.9675, 0.9727, 0.976, 0.9744, 0.9747, 0.9748]
# for i in hidden_nodes_array:
#     hidden_nodes_array_succeed.append(BPNetDemo.performance(i, 0.1, 5))
# print(hidden_nodes_array_succeed)
plt.plot(hidden_nodes_array, hidden_nodes_array_succeed, marker='o', mec='r', mfc='w',label=u'hidden_nodes_relation')
plt.legend()  # 让图例生效
plt.xticks(hidden_nodes_array, hidden_nodes_array, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"hidden_nodes") #X轴标签
plt.ylabel("succeed_rate") #Y轴标签
plt.title("A hidden_nodes plot") #标题

plt.show()