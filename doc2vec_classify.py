#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

import sys
import os
import pdb
import time
import gensim
import shutil 

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


def classify_prepare(needClassifyName):
    #初始化文件夹
    shutil.rmtree('1.train')  
    os.mkdir('1.train') 

    """
    数据预处理
    filepath            需要做分类数据集路径
    filename            需要做分类数据集名称
    file_train_pos      需要做分类数据的训练集保存路径
    file_train_neg      其它类别数据的训练集保存路径
    """
    filepath = 'result/' + str(needClassifyName) + '.txt'
    filename = filepath.split('/')[0:]
    file_train_pos = open(str('1.train/1.'+filename[-1]),'w')
    file_train_neg = open('1.train/2.其它类别.txt','w')

    #打开filepath文件
    myfile = open(filepath) 
    #获取文件总行数
    linesNum = len(myfile.readlines()) 
    #print "There are %d lines in %s" % (linesNum, filepath) 

    trainNum = linesNum

    #打开filepath文件，按行读取，写入file_train_pos里面
    for index, line in enumerate(open(filepath)):
        file_train_pos.writelines(line)

    """
    文件夹遍历回调函数
    args                 需要做分类数据集名称
    meragefiledir        文件夹路径
    filenames            所有数据集名称数组
    """
    def processDirectoryprepare ( args, meragefiledir, filenames ):
        #获取总文件数
        filesNum = len(filenames)
        #计算每类文件所需要的提供的训练集数目
        each_train_Num = int(1.25*trainNum/filesNum)
        for filename in filenames:
           #不读取需要做分类的数据集
           if(filename!=args):
                filetemp = meragefiledir +'/'+ filename
                for index, line in enumerate(open(filetemp)):
                    #每类最多读取each_test_Num作为测试集
                    if(index<each_train_Num):
                        file_train_neg.writelines(line)
                    #超过each_train_Num跳出循环
                    else:
                        break

    """
    读取result文件夹里面所有类别文件名称
    result_filedir       所有分类数据文件夹路径
    processDirectory     回调函数
    filename             需要做分类数据集名称
    """
    result_filedir = utils.to_unicode(os.getcwd()+'/result')
    os.path.walk(result_filedir, processDirectoryprepare, filename )

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
	    string_save = test_pinyin.get_pinyin(result[0],'')
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
x_unsup     需要预测的数据集
y_train     训练集标签
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

    #遍历3.unsup目录下所有文件，储存到全局变量sources和totalNum内
    unsup_filedir = utils.to_unicode(os.getcwd()+'/3.unsup')
    os.path.walk(unsup_filedir, processDirectory, [sources,totalNum] )
    
    #测试集的LabeledSentence构建
    sentences_unsup = LabeledLineSentence(sources)
    x_unsup = sentences_unsup
    
    print '-------------------------------------------'
    print 'all sources path:'
    print sources
    print totalNum
    print '-------------------------------------------'

    #重置sources，totalNum
    sources = {}
    totalNum = {}

    #遍历result目录下所有文件，储存到全局变量sources和totalNum内
    result_filedir = utils.to_unicode(os.getcwd()+'/result')
    os.path.walk(result_filedir, processDirectory, [sources,totalNum] )
    
    #测试集的LabeledSentence构建
    sentences_result = LabeledLineSentence(sources)
    x_result = sentences_result
    
    print '-------------------------------------------'
    print 'all sources path:'
    print sources
    print totalNum
    print '-------------------------------------------'

    return x_train,y_train,x_unsup,x_result

"""
训练函数
()内为默认值
x_train         训练集
x_unsup         需要预测的集合
size(400)       向量尺寸
epoch_num(10)   训练迭代次数
"""
def train(x_train,x_unsup,x_result,size = 400,epoch_num=10):
    
    print 'Model training parameters: min count=1  window=10 size='+str(size)+' sample=1e-3 negative=5 workers=3'
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)
    
    print 'start building vocabulary...'
    model_dm.build_vocab(x_train.to_array()+x_unsup.to_array())
    model_dbow.build_vocab(x_train.to_array()+x_unsup.to_array())
    print 'finish building vocabulary...'
    
    print "start training train's dataset..."
    all_train_reviews = x_train.to_array()+x_result.to_array()
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
    x_unsup_reviews = x_unsup.to_array()
    j = '#'
    if __name__ == '__main__':
        for epoch in range(epoch_num):
            j += '#'
            sys.stdout.write(str(int((epoch/epoch_num)*100))+'%  ||'+j+'->'+"\r")
            sys.stdout.flush()
            #将序列x_unsup_reviews的所有元素随机排序
            shuffle(x_unsup_reviews)
            model_dm.train(x_unsup_reviews)
            model_dbow.train(x_unsup_reviews)
    print

    """
    print "saving unsup's model..."        
    fname_dm_unsup = 'model/model.dm.unsup'
    fname_dbow_unsup = 'model/model.dbow.unsup'
    model_dm.save(fname_dm_unsup)
    model_dbow.save(fname_dbow_unsup)
    """

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

    unsup_vecs_dm = getVecs(model_dm, x_unsup.to_array(), size)
    unsup_vecs_dbow = getVecs(model_dbow, x_unsup.to_array(), size)
    unsup_vecs = np.hstack((unsup_vecs_dm, unsup_vecs_dbow))

    return train_vecs,unsup_vecs

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
训练测试集得到向量预测
train_vecs   训练集向量
y_train      训练数据标签
unsup_vecs   预测集向量
model_dm     dm训练模型
model_dbow   dbow训练模型
"""
def Classifier_predict(RF,train_vecs,y_train,unsup_vecs, model_dm, model_dbow):
  
    #Build a forest of trees from the training set (train_vecs, y_train).
    RF.fit(train_vecs, y_train)  

    #Predict class probabilities for unsup_vecs.
    pred_probas = RF.predict_proba(unsup_vecs)[:,1]

    #保存训练测试集预测结果到'outcomes/train_predicting_outcomes.txt'
    file_path = 'outcomes/train_predicting_outcomes.txt'
    writeToTxt(pred_probas,file_path)

    return RF

"""
主函数
size       转化向量维度
epoch_num  训练迭代次数
"""
if __name__ == "__main__":
    size =400
    epoch_num = 10

    print sys.argv[1]
    #用于分类器的训练集准备
    classify_prepare(sys.argv[1])

    print 'start get dataset...'
    #用于训练的数据获取
    x_train,y_train,x_unsup,x_result = get_dataset()
    print 'finish get dataset...'

    print 'start train...'
    #对数据进行训练
    model_dm,model_dbow = train(x_train,x_unsup,x_result,size,epoch_num)
    print 'finish train...'

    #如果已经有训练好的数据集就直接读取
    #model_dm = Doc2Vec.load('model/model.dm.unsup')
    #model_dbow = Doc2Vec.load('model/model.dbow.unsup')

    print 'start get vector of sentences...'
    #得到训练数据集和测试数据集的向量
    train_vecs,unsup_vecs = get_vectors(model_dm,model_dbow)
    print 'finish get vector of sentences...'

    print 'start classifier...'
    #这里使用的是Random Forest分类器，其它分类器也可以选择
    #训练速度:SGD > SVM > RandomForest > GBDT
    #准确率:GBDT ~= RandomForest > SVM ~= SGD
    from sklearn.ensemble import RandomForestClassifier  
    RF = RandomForestClassifier(n_estimators=1200,max_depth=14,class_weight={0:0.3,1:0.7})

    RF=Classifier_RF(RF, train_vecs, y_train, unsup_vecs, model_dm, model_dbow)
    print 'finish classifier...'
