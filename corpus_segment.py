# python3.6

# 中文语料切分

import os
import jieba
import re
import numpy as np
from gensim.models import Word2Vec
import pickle



def read_file(path,encoding='utf8'):   #从文件中读取数据
    with open(path,'r',encoding=encoding,errors='ignore') as fp:
        content = fp.read()
    return(content)

def save_file(save_path,content,encoding='utf8'):   #文件保存为utf8格式
    with open(save_path,'w',encoding=encoding) as fp:
        fp.write(content)

def corpus_segment(corpus_path,segment_path):  #对目录下中文文本进行分词处理
    if not os.path.exists(segment_path):
        os.makedirs(segment_path)
    dir_list = os.listdir(corpus_path)
    for mydir in dir_list:
        class_path_raw = os.path.join(corpus_path,mydir)
        class_path_need = os.path.join(segment_path,mydir)
        if not os.path.exists(class_path_need):
            os.makedirs(class_path_need)
        file_list = os.listdir(class_path_raw)
        for myfile in file_list:
            content = read_file(os.path.join(class_path_raw,myfile),encoding='gbk')
            content = re.sub(r'\n|\s+','',content)
            content_seg = jieba.cut(content)
            save_path = os.path.join(class_path_need,myfile)
            save_file(save_path,' '.join(content_seg))


class MySentences(object):  #为训练word2vec 模型提供拆分后的数据类
    def __init__(self,dirname):
        self.dirname = dirname

    def __iter__(self):
        for root,dirs,files in os.walk(self.dirname):
            for filename in files:
                file_path = os.path.join(root,filename)
                if(re.search(r'\.txt$',file_path)):
                    content = read_file(file_path)
                    yield (re.split(r'\s+',content))

def segment_matrix(segment_path,matrix_path,model):  #把切分好的词向量通过词矩阵转换为词矩阵
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path)
    dir_list = os.listdir(segment_path)

    #    temp
    for mydir in dir_list:
        class_path_raw = os.path.join(segment_path,mydir)
        class_path_need = os.path.join(matrix_path,mydir)
        if not os.path.exists(class_path_need):
            os.makedirs(class_path_need)
        file_list = os.listdir(class_path_raw)
        for myfile in file_list:
            content = read_file(os.path.join(class_path_raw,myfile))
            content = re.split(r'\s+',content)
            content_matrix = np.array([model[word] for word in content if word in model.wv.vocab.keys()])
            save_path = os.path.join(class_path_need,(myfile+'.dat'))
            with open(save_path,'wb') as fp:
                pickle.dump(content_matrix,fp)



if __name__ == '__main__':
    '''
    train_corpus_path = './data/train_corpus'
    test_corpus_path = './data/test_corpus'
    train_corpus_seg_path = './data/train_corpus_seg'
    test_corpus_seg_path = './data/test_corpus_seg'
    corpus_segment(train_corpus_path,train_corpus_seg_path)
    corpus_segment(test_corpus_path,test_corpus_seg_path)

    train_corpus_seg_path = './data/train_corpus_seg'

    sentences = MySentences(train_corpus_seg_path)

    model_word2vec = Word2Vec(sentences,size=200,window=10,min_count=10)
    model_word2vec.save('data/model/word2vec')
    '''
    train_corpus_seg_path = '../data/THUCNews/train_corpus_seg'
    test_corpus_seg_path = '../data/THUCNews/test_corpus_seg'

    train_corpus_matrix_path = '../data/THUCNews/train_corpus_matrix'
    test_corpus_matrix_path = '../data/THUCNews/test_corpus_matrix'

    model = Word2Vec.load('data/model/word2vec')
    segment_matrix(train_corpus_seg_path, train_corpus_matrix_path, model)
    segment_matrix(test_corpus_seg_path,test_corpus_matrix_path,model)










