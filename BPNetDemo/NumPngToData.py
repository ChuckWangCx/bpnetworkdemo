import scipy.misc
import numpy
# S型激活函数引入
import scipy.special
# 矩阵运算
import matplotlib.pyplot
# 对象序列化
import shelve


# 检验神经网络
def check(inputs_list, wih, who):
    # 转换成二维矩阵并转置
    inputs = numpy.array(inputs_list, ndmin=2).T

    # 隐藏层输入
    hidden_inputs = numpy.dot(wih, inputs)
    # 隐藏层输出
    hidden_outputs = scipy.special.expit(hidden_inputs)
    # 输出层输入
    final_inputs = numpy.dot(who, hidden_outputs)
    # 输出层输出
    final_outputs = scipy.special.expit(final_inputs)

    return final_outputs

if __name__ == '__main__':
    image_file_name = "C:/Users/DELL/Desktop/Python神经网络/111.png"
    # 将图片转化为浮点数组
    img_array = scipy.misc.imread(image_file_name, flatten = True )
    # 匹配mnist数据格式
    img_data = 255.0 - img_array.reshape(784)
    # 归一化
    img_data = (img_data / 255.0 * 0.99) + 0.01
    dbase = shelve.open("C:/Users/DELL/Desktop/Python神经网络/network")
    wih = dbase['wih']
    who = dbase['who']
    dbase.close()
    label = numpy.argmax(check(img_data, wih, who))
    # 输出结果
    print(label)
