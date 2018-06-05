import re
import pymysql
import os
import jieba
from collections import Counter



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
    for id in range(67900):
        try:
            if(id%300==0):
                print(id)
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






if __name__ =='__main__':
    path_raw_train = 'C:\\projects\\data\\CNEWS\\train'
    path_raw_test = 'C:\\projects\\data\\CNEWS\\test'


    conn = pymysql.connect(host = 'localhost',user = 'root',passwd = 'sasa',db = 'text_clf',charset='utf8')
    cur = conn.cursor()
    #create_table(cur)
    #raw_seg(raw_path, cur)
    #create_vocab(cur)

    try:
        #raw_seg(path_raw_train,cur,'tb_train_text')
        #raw_seg(path_raw_test,cur,'tb_test_text')
        create_vocab(cur)
        conn.commit()
    except:
        conn.rollback()
    finally:
        conn.close()

