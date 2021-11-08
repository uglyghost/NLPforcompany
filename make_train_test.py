#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import os
from gensim import utils  
import shutil 
 

"""
数据预处理
filepath            需要做分类数据集路径
filename            需要做分类数据集名称
file_train_pos      需要做分类数据的训练集保存路径
file_test_pos       需要做分类数据的测试集保存路径
file_train_neg      其它类别数据的训练集保存路径
file_teat_neg       其它类别数据的测试集保存路径
"""
#清空文件夹
shutil.rmtree('1.train')  
os.mkdir('1.train') 
shutil.rmtree('2.test')  
os.mkdir('2.test') 

filepath = 'result/亏损盈利.txt'
filename = filepath.split('/')[0:]
file_train_pos = open(str('1.train/1.'+filename[-1]),'w')
file_test_pos = open(str('2.test/1.'+filename[-1]),'w')
file_train_neg = open('1.train/2.其它类别.txt','w')
file_test_neg = open('2.test/2.其它类别.txt','w')

#打开filepath文件
myfile = open(filepath) 
#获取文件总行数
linesNum = len(myfile.readlines()) 
#print "There are %d lines in %s" % (linesNum, filepath) 

#按照2：8划分测试集和训练集
trainNum = int(0.8*linesNum)
testNum = linesNum - trainNum

#打开filepath文件，按行读取
for index, line in enumerate(open(filepath)):
    #读取前trainNum行作为训练集
    if (index < trainNum):
        file_train_pos.writelines(line)
    #读取后面testNum行作为测试集
    else:
        file_test_pos.writelines(line)
"""
文件夹遍历回调函数
args                 需要做分类数据集名称
meragefiledir        文件夹路径
filenames            所有数据集名称数组
"""
def processDirectory ( args, meragefiledir, filenames ):
    #获取总文件数
    filesNum = len(filenames)
    #计算每类文件所需要的提供的训练集数目
    each_train_Num = int(1.25*trainNum/filesNum)
    #计算每类文件所需要的提供的测试集数目
    each_test_Num = int(testNum/filesNum)
    for filename in filenames:
       #不读取需要做分类的数据集
       if(filename!=args):
            filetemp = meragefiledir +'/'+ filename
            for index, line in enumerate(open(filetemp)):
                #每类最多读取each_test_Num作为测试集
                if(index<each_test_Num):
                    file_train_neg.writelines(line)
                    file_test_neg.writelines(line)
                #每类最多读取each_train_Num作为训练集
                elif(index<each_train_Num):
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
os.path.walk(result_filedir, processDirectory, filename )
