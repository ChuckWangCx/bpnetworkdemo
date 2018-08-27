import numpy
# S型激活函数引入
import scipy.special
# 矩阵运算
import matplotlib.pyplot
# import shelve

# 神经网络class定义
class neuralNetwork :
    # 初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化神经网络输入层，隐藏层，输出层节点
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 初始化输入层到隐藏层，隐藏层到输出层的权重
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 学习率
        self.lr = learningrate

        # S型激活函数
        self.activation_function = lambda x : scipy.special.expit(x)

        pass

    # 训练函数
    def train(self, input_list, targets_list):
        # 转换成二维矩阵并转置
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层输出（最终输出）
        final_outputs = self.activation_function(final_inputs)

        # 输出层误差
        output_errors = targets - final_outputs
        # 隐藏层误差
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 隐含层到输出层权重调整
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # 输入层到隐藏层权重调整
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    # 检验神经网络
    def check(self, inputs_list):
        # 转换成二维矩阵并转置
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层输出
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# 性能
def performance(hidden_nodes, learning_rate, epochs) :
    # 设置各层节点
    input_nodes = 784
    # hidden_nodes = 200
    output_nodes = 10

    # 学习率
    # learning_rate = 0.1

    # 创建神经网络模型
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取训练数据
    training_data_file = open("C:/Users/DELL/Desktop/Python神经网络/mnist_train.csv", 'r')
    train_data_list = training_data_file.readlines()
    training_data_file.close()

    # 迭代5次
    # epochs = 5

    # 训练模型
    for e in range(epochs):
        # 循环读取记录
        for record in train_data_list:
            # 解析数据
            all_values = record.split(',')
            # 归一化处理（标准化处理）
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        pass

    # 读取测试数据
    test_data_file = open("C:/Users/DELL/Desktop/Python神经网络/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 测试结果数据
    scorecard = []

    # 测试模型
    for record in test_data_list:
        # 解析数据
        all_values = record.split(',')
        # 正确结果
        correct_label = int(all_values[0])
        # 输入数据
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 输出数据
        outputs = n.check(inputs)
        # 取输出节点中的最大值
        label = numpy.argmax(outputs)
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    scorecard_array = numpy.asarray(scorecard)
    return scorecard_array.sum() / scorecard_array.size
    # print("正确率：", scorecard_array.sum() / scorecard_array.size)

# db = shelve.open("C:/Users/DELL/Desktop/Python神经网络/compare/200-")
# db['wih']  = n.wih
# db['who']  = n.who
# db.close( )

