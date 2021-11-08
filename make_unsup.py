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
shutil.rmtree('3.unsup')  
os.mkdir('3.unsup') 
shutil.rmtree('outcomes')  
os.mkdir('outcomes') 

filepath = 'result/亏损盈利.txt'
filename = filepath.split('/')[0:]
file_unsup = open(str('3.unsup/1.预测数据集.txt'),'w')
file_result = open(str('outcomes/true_outcomes.txt'),'w')

"""
文件夹遍历回调函数
args                 需要做分类数据集名称
meragefiledir        文件夹路径
filenames            所有数据集名称数组
"""
def processDirectory ( args, meragefiledir, filenames ):
    #计算每类文件所需要的提供的训练集数目
    each_type_Num = 1000
    for filename in filenames:
       #不读取需要做分类的数据集
       filetemp = meragefiledir +'/'+ filename
       if(filename!=args):
            for index, line in enumerate(open(filetemp)):
                if(index>each_type_Num):
                    break
                file_unsup.writelines(line)
                file_result.writelines('0')
                file_result.writelines('\n')
       else:
            for index, line in enumerate(open(filetemp)):
                if(index>each_type_Num):
                    break
                file_unsup.writelines(line)
                file_result.writelines('1')
                file_result.writelines('\n')

"""
读取result文件夹里面所有类别文件名称
result_filedir       所有分类数据文件夹路径
processDirectory     回调函数
filename             需要做分类数据集名称
"""
result_filedir = utils.to_unicode(os.getcwd()+'/result')
os.path.walk(result_filedir, processDirectory, filename )
