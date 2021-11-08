#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 20:10:36 2017

@author: wang

1.数据文件合并
2.标点符号清洗
3.中文文档分词
"""
import os 
import jieba

"""
merage file 
对中文文档进行分词处理；
并清洗文档内标点符号；
最后合并文档到特定名称的文件夹。

filedir     合并文件的文件夹
filenames   合并文件名称数组
file        合并后存储文件名
"""
def meragefile(filedir,filenames,file):
    #先遍历文件名  
    for filename in filenames:  
        filepath=filedir+'/'+filename  
        #遍历单个文件，读取行数  
        for line in open(filepath):
            #使用jieba分词
	    newline = jieba.cut(line, cut_all=False)
            #去除内部的标点符号
            str_out = ' '.join(newline).encode('utf-8').replace('，','').replace('。','').replace('?','').replace('!','')\
                         .replace('“','').replace('”','').replace('：','').replace('‘','').replace('’','').replace('-','')\
                         .replace('（','').replace('）','').replace('《','').replace('》','').replace('：','').replace('。','')\
                         .replace('、','').replace('...','').replace('.','').replace(',','').replace('？','').replace('！','')\
                         .replace('=','').replace('+','').replace('【','').replace('】','').replace('/','').replace('(','')\
                         .replace(')','').replace('%','').replace('\n','').replace('<br />','').replace('.','').replace('.','')\
                         .replace('~','').replace('*','').replace(':','').replace(';','').replace('#','').replace('—','')\
			 .replace('〔','').replace('〕','').replace('·','').replace('&','')
            file.writelines(str_out)  
            #file.write('\n')
        #读取完一则消息后换行
        file.write('\n')  
    #关闭文件  
    file.close()
    return

"""
processDirectory
处理得到得到
1.需合并文件名称；
2.合并文件路径；
3.合并文件后文件的名称
带入meragefile函数操作
args         传递给回调函数的元组
filedir      需要遍历的目录路径
filenames    需要遍历目录的文件数组 
"""
def processDirectory ( args, filedir, filenames ):
    #读取文件夹名称，并打印
    print 'Directory',filedir
    #读取文档中文件名称，并打印
    for filename in filenames:
        print ' File',filename
        
    #对文件名进行对'.'的分割操作，获取文件名后缀
    suffix = os.path.splitext(filename)[-1]
    #判断是否为txt文件
    if(str(suffix) == '.txt'):
        #对目录进行对'/'的分割，获取结果存到result
        result = filedir.split('/')[1:]
        #保存文件名为"目录名称.txt"
        changefilename = os.getcwd() + '/result/' + result[-1]+'.txt'
        #将"目录名称.txt"作为合并文件txt的名称
        merage_filename = open(changefilename,'w') 
        meragefile(filedir,filenames,merage_filename)

#获取当前目录下需要处理的文件夹名称
original_dataset_filedir = os.getcwd()+'/Dataset'
"""
original_dataset_filedir 需要遍历的目录路径
processDirectory         回调函数，遍历路径进行处理的函数
None                     传递给回调函数的元组
"""
os.path.walk(original_dataset_filedir, processDirectory, None )
