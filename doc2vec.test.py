#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

import sys
import os
import pdb
import time
import gensim
import numpy as np
import sklearn.metrics as metrics

from gensim import utils  
from gensim.models.doc2vec import LabeledSentence  
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from sklearn.cross_validation import train_test_split
from random import shuffle  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix  
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from xpinyin import Pinyin

#将文字转化为拼音便于标识
test_pinyin = Pinyin()

LabeledSentence = gensim.models.doc2vec.LabeledSentence

#LabeledSentence构建
class LabeledLineSentence(object):  
    def __init__(self, sources):  
        self.sources = sources  
  
        flipped = {}  
  
        # make sure that keys are unique  
        for key, value in sources.items():  
            if value not in flipped:  
                flipped[value] = [key]  
            else:  
                raise Exception('Non-unique prefix encountered')  
  
    def __iter__(self):  
        for source, prefix in self.sources.items():  
            with utils.smart_open(source) as fin:  
                for item_no, line in enumerate(fin):  
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])  
  
    def to_array(self):  
        self.sentences = []  
        for source, prefix in self.sources.items():  
            with utils.smart_open(source) as fin:  
                for item_no, line in enumerate(fin):  
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))  
        return self.sentences  
  
    def sentences_perm(self):  
        shuffle(self.sentences)  
        return self.sentences 

"""
回调函数
args                 [sources,totalNum]
meragefiledir        文件夹路径
filenames            目录下所有文件的名称数组
"""
def processDirectory ( args, meragefiledir, filenames ):
    for index, filename in enumerate(filenames):
        #得到文件名称
        result = filename.split('.')[0:]
        #判断文件是否是txt
        if(result[-1]=='txt'):
            #得到文件名称的拼音
	    string_save = test_pinyin.get_pinyin(result[1],'')
            #对文件路径进行unicode编码(不进行编码中文路径会出错)
            dir_filename = utils.to_unicode(meragefiledir+'/'+filename)
            #保存路径集和标签到sources对象
            args[0][dir_filename] = str(string_save)
            #保存目录下各数据集行数到totalNum对象
            args[1][index] = len(open(dir_filename).readlines())
            print '-------------------------------------------'
            print 'category No: ' + str(index)
            print 'category name: ' + str(string_save)
	    print 'category number: ' + str(args[1][index])
            #print '-------------------------------------------'

"""
get_dataset
读取训练集和测试集
x_train     训练集数据
x_test      测试集数据
y_train     训练集标签
y_test      测试集标签
"""
def get_dataset():
    #用于储存路径集标签的对象
    sources = {}
    #用于储存数据集行数的对象
    totalNum = {}

    #遍历1.train目录下所有文件，储存到全局变量sources和totalNum内
    train_filedir = utils.to_unicode(os.getcwd()+'/1.train')
    os.path.walk(train_filedir, processDirectory, [sources,totalNum] )
    
    print '-------------------------------------------'
    print 'all sources path:'
    print sources
    print totalNum

    #训练集的LabeledSentence构建
    sentences_train = LabeledLineSentence(sources)
    x_train = sentences_train

    #y_train构建
    for epoch in range(len(totalNum)):
        if ( epoch==0 ):
            y_train = epoch*np.ones(totalNum[epoch])
        else:
            y_train = np.concatenate((y_train, epoch*np.ones(totalNum[epoch])))

    #重置sources，totalNum
    sources = {}
    totalNum = {}

    #遍历2.test目录下所有文件，储存到全局变量sources和totalNum内
    train_filedir = utils.to_unicode(os.getcwd()+'/2.test')
    os.path.walk(train_filedir, processDirectory, [sources,totalNum] )
    
    #测试集的LabeledSentence构建
    sentences_test = LabeledLineSentence(sources)
    x_test = sentences_test
    
    print '-------------------------------------------'
    print 'all sources path:'
    print sources
    print totalNum
    print '-------------------------------------------'

    #y_test构建
    for epoch in range(len(totalNum)):
        if ( epoch==0 ):
            y_test = epoch*np.ones(totalNum[epoch])
        else:
            y_test = np.concatenate((y_test, epoch*np.ones(totalNum[epoch])))

    #print str(len(x_test))  + " and "  + str(len(y_test))
    #x_train, x_test, y_train, y_test = train_test_split(sentences.to_array(), y, test_size=0.2)
    #pdb.set_trace()
    #x_train = cleanText(x_train)
    #x_test = cleanText(x_test)

    return x_train,x_test,y_train,y_test

"""
训练函数
()内为默认值
x_train         训练集
x_test          测试集
size(400)       向量尺寸
epoch_num(10)   训练迭代次数
"""
def train(x_train,x_test,size = 400,epoch_num=10):
    
    print 'Model training parameters: min count=1  window=10 size='+str(size)+' sample=1e-3 negative=5 workers=3'
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)
    
    print 'start building vocabulary...'
    model_dm.build_vocab(x_train.to_array()+x_test.to_array())
    model_dbow.build_vocab(x_train.to_array()+x_test.to_array())
    print 'finish building vocabulary...'
    
    print "start training train's dataset..."
    all_train_reviews = x_train.to_array()
    j = '#'
    if __name__ == '__main__':
        for epoch in range(epoch_num):
            j += '#'
            sys.stdout.write(str(int((epoch/epoch_num)*100))+'%  ||'+j+'->'+"\r")
            sys.stdout.flush()
            #将序列all_train_reviews的所有元素随机排序
            shuffle(all_train_reviews)
            model_dm.train(all_train_reviews)
            model_dbow.train(all_train_reviews)
    print

    print "saving train's model..."
    #fname_dm_train = 'model/model.dm.train'
    #fname_dbow_train = 'model/model.dbow.train'
    #model_dm.save(fname_dm_train)
    #model_dbow.save(fname_dbow_train)

    print "start training test's dataset..."
    x_test_reviews = x_test.to_array()
    j = '#'
    if __name__ == '__main__':
        for epoch in range(epoch_num):
            j += '#'
            sys.stdout.write(str(int((epoch/epoch_num)*100))+'%  ||'+j+'->'+"\r")
            sys.stdout.flush()
            #将序列x_test_reviews的所有元素随机排序
            shuffle(x_test_reviews)
            model_dm.train(x_test_reviews)
            model_dbow.train(x_test_reviews)
    print

    print "saving test's model..."        
    fname_dm_test = 'model/model.dm.test'
    fname_dbow_test = 'model/model.dbow.test'
    model_dm.save(fname_dm_test)
    model_dbow.save(fname_dbow_test)

    return model_dm,model_dbow

"""
得到预料向量的功能函数
model   训练模型
corpus  语料
size    向量大小
"""
def getVecs(model, corpus, size):
    #在模型model中按照corpus的顺序读取训练结果，并转化为尺寸等于size的向量
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

"""
将训练数据转化为可以判断的向量
model_dm       dm训练得到的模型
model_dbow     dbow训练得到的模型
"""
def get_vectors(model_dm,model_dbow):

    #得到x_train dm模型的向量train_vecs_dm
    train_vecs_dm = getVecs(model_dm, x_train.to_array(), size)
    #得到x_train dbow模型的向量train_vecs_dbow
    train_vecs_dbow = getVecs(model_dbow, x_train.to_array(), size)
    #将dm模型和dbow模型得到的预测向量合并
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

    test_vecs_dm = getVecs(model_dm, x_test.to_array(), size)
    test_vecs_dbow = getVecs(model_dbow, x_test.to_array(), size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs,test_vecs

"""
"""
def Classifier_lr(train_vecs,y_train,test_vecs, y_test):
    
    from sklearn.linear_model import SGDClassifier

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    print 'lr:'
    print 'Train Accuracy: %.4f'%lr.score(train_vecs, y_train)
    print 'Test Accuracy: %.4f'%lr.score(test_vecs, y_test)

    return lr

"""
"""
def Classifier_svm(train_vecs,y_train,test_vecs, y_test):
    
    from sklearn.svm import SVC
    svm = SVC()

    svm.fit(train_vecs, y_train)

    print 'svm:'
    print 'Train Accuracy: %.4f'%lr.score(train_vecs, y_train)
    print 'Test Accuracy: %.4f'%lr.score(test_vecs, y_test)

    return svm
"""
"""
def Classifier_GBDT(train_vecs,y_train,test_vecs, y_test):

    from sklearn.ensemble import GradientBoostingClassifier  
    GBDT = GradientBoostingClassifier(n_estimators=1000,max_depth=14)  
    GBDT.fit(train_vecs, y_train) 

    print 'GBDT:'
    print 'Test Accuracy: %.4f'%GBDT.score(test_vecs, y_test)

    return GBDT

"""
分类功能函数
获取目录下文件名称数组
"""
def GetFileList(dir, fileList):
    newDir = dir
    #判断是非是文件
    if os.path.isfile(dir):
        #是文件增加
        fileList.append(dir)
    #判断是否是文件夹
    elif os.path.isdir(dir):  
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            #if s == "xxx":
                #continue
            #递归调用，遍历所有文件
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)  
    
    #根据名称排序，以免出现顺序问题
    if (len(fileList)>0):  
        fileList.sort() 

    return fileList

"""
分类功能函数
将结果写入到txt文件
"""
def writeToTxt(list_name,file_path):
    try:
        fp = open(file_path,"w+")
        for item in list_name:
            fp.write(str(item)+"\n")
        fp.close()
    except IOError:
        print("fail to open file")

"""
利用随机森林(random forest)分类器对策是数据进行分类
两种分类方式：
1.训练测试集得到向量预测
2.直接通过测试集内容预测
train_vecs   训练集向量
y_train      训练数据标签
test_vecs    测试集向量
y_test       测试数据标签
model_dm     dm训练模型
model_dbow   dbow训练模型
"""
def Classifier_predict_RF(train_vecs,y_train,test_vecs, y_test, model_dm, model_dbow):

    from sklearn.ensemble import RandomForestClassifier  
    RF = RandomForestClassifier(n_estimators=1200,max_depth=14,class_weight={0:0.3,1:0.7})  
    #Build a forest of trees from the training set (train_vecs, y_train).
    RF.fit(train_vecs, y_train)  

    print 'RF:'
    print 'Test Accuracy: %.4f'%RF.score(test_vecs, y_test)

    #Predict class probabilities for test_vecs.
    pred_probas = RF.predict_proba(test_vecs)[:,1]

    #读取测试及文件路径
    test_filedir = utils.to_unicode(os.getcwd()+'/2.test')
    filelist = GetFileList(test_filedir, [])

    print "test flie path array:"
    print filelist

    #保存dm模型的测试集向量数组
    test_arrays_dm = []
    #保存dbow模型的测试集向量数组
    test_arrays_dbow = []
    #保存真实的标签
    true_labels=[]  
    a=open(filelist[0])  
    b=open(filelist[1])  
    test_content1=a.readlines()  
    test_content2=b.readlines()  
    for i in test_content1:  
        test_arrays_dm.append(model_dm.infer_vector(i)) 
        test_arrays_dbow.append(model_dbow.infer_vector(i)) 
        true_labels.append(1)  
    for i in test_content2:  
        test_arrays_dm.append(model_dm.infer_vector(i))  
        test_arrays_dbow.append(model_dbow.infer_vector(i))  
        true_labels.append(0)  

    #将dm模型和dbow模型得到的预测向量合并
    test_vecs_predicting = np.hstack((test_arrays_dm, test_arrays_dbow))
 
    #用于保存RF直接预测结果的数组
    test_labels_RF=[]  

    j = '#'
    if __name__ == '__main__':
        for i in range(len(test_arrays_dm)):  
            if(((20*i)%len(test_arrays_dm))==0):
                j += '#'
            sys.stdout.write(str(int((i/len(test_arrays_dm))*100))+'%  ||'+j+'->'+"\r")
            sys.stdout.flush()
            xpredictnd = np.array(test_vecs_predicting[i]).reshape(1,-1) 
            test_labels_RF.append(RF.predict(xpredictnd)) 
    print 

    #输出直接预测的准确率，并输出预测矩阵
    print("RF:")  
    print(metrics.accuracy_score(test_labels_RF,true_labels))  
    print(confusion_matrix(test_labels_RF,true_labels)) 

    #保存训练测试集预测结果到'outcomes/train_predicting_outcomes.txt'
    file_path = 'outcomes_test/train_predicting_outcomes.txt'
    writeToTxt(pred_probas,file_path)
    #保存真是标签到'outcomes/true_outcomes.txt'
    file_path = 'outcomes_test/true_outcomes.txt'
    writeToTxt(y_test,file_path)
    #保存直接预测结果到'outcomes/predicting_outcomes.txt'
    file_path = 'outcomes_test/predicting_outcomes.txt'
    writeToTxt(test_labels_RF,file_path)

    return RF

#ROC绘图函数
def ROC_curve(lr,y_test):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(test_vecs)[:,1]

    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()

#主函数
if __name__ == "__main__":
    size =400
    epoch_num = 10
    print 'start get dataset...'
    x_train,x_test,y_train,y_test = get_dataset()
    print 'finish get dataset...'
    print 'start train...'
    #model_dm,model_dbow = train(x_train,x_test,size,epoch_num)
    print 'finish train...'
    print 'start get vector of sentences...'
    #如果已经有训练好的数据集就直接读取
    model_dm = Doc2Vec.load('model/model.dm.test')
    model_dbow = Doc2Vec.load('model/model.dbow.test')
    train_vecs,test_vecs = get_vectors(model_dm,model_dbow)
    print 'finish get vector of sentences...'
    print 'start classifier...'
    #lr=Classifier_lr(train_vecs,y_train,test_vecs, y_test)
    #svm=Classifier_svm(train_vecs,y_train,test_vecs, y_test)
    #GBDT=Classifier_GBDT(train_vecs,y_train,test_vecs, y_test)
    RF=Classifier_predict_RF(train_vecs, y_train, test_vecs, y_test, model_dm, model_dbow)
    print 'finish classifier...'
    ROC_curve(RF,y_test)
