from d2lzh import d2lzh as d2l
from mxnet.gluon import data as gdata
from mxnet import autograd,nd
import sys
import time

#实现softmax运算
def softmax(X):
    X_exp=X.exp()
    partition=X_exp.sum(axis=1, keepdims=True)
    return X_exp/partition     #这里应用了广播机制

#定义模型
def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)),W ) +b )

#定义损失函数
def cross_entropy(y_hat,y):
    return -nd.pick(y_hat,y).log()

#计算分类准确率
def accuracy(y_hat,y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar  #返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同

def evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        y=y.astype('float32')
        acc_sum += (net(X).argmax(axis=1)==y).sum().asscalar()
        n += y.size
    return acc_sum/n

#训练模型
def train_ch3(net, train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,trainer=None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            with autograd.record():
                y_hat=net(X)
                l=loss(y_hat,y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params,lr,batch_size)
            else:
                trainer.step(batch_size)
            y=y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1)==y).sum().asscalar()
            n += y.size
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch+1,train_l_sum/n,train_acc_sum/n,test_acc) )




#读取数据集
batch_size=256    #设置批量大小为256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

#初始化模型参数
num_inputs=784   #模型的输入向量的长度为28x28=784，该向量的每个元素对应图像中每个像素。
num_outputs=10    #图像有10个类别，单层神经网络输出层的输出个数为10

W=nd.random.normal(scale=0.01,shape=(num_inputs,num_outputs))
b=nd.zeros(num_outputs)

#为模型参数附上梯度
W.attach_grad()
b.attach_grad()

num_epochs,lr=5,0.1

train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,batch_size,[W,b],lr)


#预测
for X,y in test_iter:
    break

true_labels=d2l.get_fashion_mnist_labels(y.asnumpy())
pred_lables=d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles=[true+'\n'+pred for true, pred in zip(true_labels,pred_lables)]
d2l.show_fashion_mnist(X[0:9],titles[0:9])



