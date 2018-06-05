import re
import pymysql
import os
import jieba
from collections import Counter
import tensorflow.contrib.keras as kr
from gensim.models import Word2Vec



def create_table(cur):
    sql1 = """
    create table if not exists tb_train_text(
    id int unsigned PRIMARY KEY Auto_Increment,
    seg text,
    label varchar(50),
    word_id varchar(20000)
    )
    """
    sql2 = """
        create table if not exists tb_test_text(
        id int unsigned PRIMARY KEY Auto_Increment,
        seg text,
        label varchar(50),
        word_id varchar(20000)
        )
        """
    sql3 = """
            create table if not exists tb_word_id(
            word varchar(50),
            id int
            )
            """
    cur.execute(sql1)
    cur.execute(sql2)
    cur.execute(sql3)



def raw_seg(raw_path,cur,table):
    class_list = os.listdir(raw_path)
    for class_name in class_list:
        class_path = os.path.join(raw_path,class_name)
        file_list = os.listdir(class_path)
        for file_name in file_list:
            file_path = os.path.join(class_path,file_name)
            with open(file_path,'r',encoding='utf8',errors='ignore')as fp:
                content = fp.read()
            content = re.sub(r'\s+|\n','',content)
            seg_content = (' '.join(list(jieba.cut(content)))[:5000]).strip() #只存储前5000个字符
            #print(seg_content)
            sql = "insert into {0}(seg,label) values(%s,%s)".format(table)
            cur.execute(sql,(seg_content,class_name))

def create_vocab(cur,vocab_size = 40000):
    vocab = Counter()
    cur.execute('select max(id) from tb_train_text')
    text_num = cur.fetchone()[0]
    for id in range(text_num-1):
        try:
            sql = 'select seg from tb_train_text where id = %s;'
            cur.execute(sql, (id + 1))
            seg = cur.fetchone()[0]
            vocab_add = Counter(re.split(r'\s+', seg))
            vocab = vocab + vocab_add
        except:
            pass
    count_pairs = vocab.most_common(vocab_size-1)
    words,_ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    word_to_id = dict(zip(words,range(len(words))))
    for key in word_to_id:
        sql = "insert into tb_word_id(word,id) values(%s,%s)"
        cur.execute(sql,(key,word_to_id[key]))

def read_vocab(cur):
    sql = 'select word,id from tb_word_id'
    cur.execute(sql)
    vocab = cur.fetchall()
    return(dict(vocab))


def word_id(cur,table,max_length = 1000):
    vocab = read_vocab(cur)
    sql = 'select max(id) from  {0}'.format(table)
    cur.execute(sql)
    max_id = cur.fetchone()[0]
    for i in range(max_id):
        id = i+1
        sql = 'select seg from {0} where id = {1}'.format(table,id)
        cur.execute(sql)
        contain = cur.fetchone()[0]
        contain_list = re.split(r'\s+', contain)
        contain_id = [vocab[x] for x in contain_list if x in vocab]
        contain_pad = kr.preprocessing.sequence.pad_sequences([contain_id], max_length)[0]
        seg_list = []
        for num_iter in contain_pad:
            seg_list.append(str(num_iter))
        seg_change = ' '.join(seg_list)
        sql_update = 'update {0} set word_id = %s where id =  %s'.format(table)
        cur.execute(sql_update,(seg_change,id))




class MySentences(object):  #为训练word2vec 模型提供拆分后的数据类
    def __init__(self,cur):
        self.cur = cur

    def __iter__(self):
        sql = 'select max(id) from tb_test_text'
        cur.execute(sql)
        max_id = cur.fetchone()[0]
        for i in range(max_id):
            id = i +1
            sql = 'select word_id from  tb_test_text  where id = {0}'.format(id)
            cur.execute(sql)
            content = cur.fetchone()[0]
            yield (re.split(r'\s+',content ))

def id_vect_model_create(cur):
    sentences = MySentences(cur)
    model_word2vec = Word2Vec(sentences, size=200, window=10, min_count=10)
    model_word2vec.save('data/model/word2vec')



if __name__ =='__main__':
    raw_path_train = 'E:\\tensor\\projects\\CNEWS\\train'
    raw_path_test = 'E:\\tensor\\projects\\CNEWS\\test'
    conn = pymysql.connect(host = 'localhost',user = 'root',passwd = 'sasa',db = 'text_clf',charset='utf8')
    cur = conn.cursor()

    id_vect_model_create(cur)

    '''
    try:
        raw_seg(raw_path_train, cur,'tb_train_text')
        raw_seg(raw_path_test,cur,'tb_test_text')
        conn.commit()
    except:
        print('!!!!!!!!!!!!!!!!!chucuo!!!!!!!!!!!!!!!!!!')
        conn.rollback()
    finally:
        conn.close()
    '''
    '''
    try:
        create_vocab(cur)
        conn.commit()
    except:
        print('!!!!!!!!!!!!!!!!!chucuo!!!!!!!!!!!!!!!!!!')
        conn.rollback()
    finally:
        conn.close()
    '''
    '''
    try:
        word_id(cur,'tb_test_text')
        conn.commit()
    except:
        print('!!!!!!!!error!!!!!!!!')
        conn.rollback()
    finally:
        conn.close()
    '''

