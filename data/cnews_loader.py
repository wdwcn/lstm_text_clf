# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

import re
from gensim.models import Word2Vec


if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育','娱乐','家居','彩票','房产','教育','时尚','时政','星座','游戏','社会','科技','股票','财经']

    categories = [native_content(x) for x in categories]

    cat_to_vec = dict(zip(categories,kr.utils.to_categorical(range(len(categories)), num_classes=len(categories))))


    return categories, cat_to_vec


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(cur,batch_size=64,source='tb_train_text'):
    categories, cat_to_vec = read_category()


    '''从mysql中生成批次数据'''
    sql_count = 'select count(*) from {0}'.format(source)
    cur.execute(sql_count)
    data_len = cur.fetchone()[0]
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len)+1)
    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i + 1) * batch_size, data_len)
        index_data = indices[start_id:end_id]
        index_data_change = "("+",".join(index_data.astype(str))+")"
        sql_data = 'select label,word_id from {0} where id in {1}'.format(source,index_data_change)
        cur.execute(sql_data)
        data_get = cur.fetchall()
        x_train = []
        y_train = []
        for row in data_get:
            x_train.append(re.split(r'\s+',row[1]))
            y_train.append(cat_to_vec[row[0]])
        yield np.array(x_train,dtype=np.int32),np.array(y_train,dtype=np.int32)



def get_embedding():
    model = Word2Vec.load('data/model/word2Vec')
    embedding = []
    for i in range(40000):
        if (str(i) not in model.wv):
            embedding.append(np.zeros(200))
        else:
            embedding.append(model.wv[str(i)])
    return(np.array(embedding,dtype=np.float32))





