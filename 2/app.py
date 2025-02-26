# -*- coding: utf-8 -*-
import streamlit as st
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from fea_extract import insert_AAC,insert_CKSAAGP,insert_CTD,insert_AAE,insert_ASDC
from sklearn.ensemble import ExtraTreesClassifier
seed = 10

st.title("PAVP预测网站")
st.markdown('''
主要针对抗冠状病毒科、逆转录病毒科、疱疹病毒科、副粘病毒科、正粘病毒科、黄病毒科病毒的抗病毒肽
''')
#获得序列
text = st.text_area(label = '请输入fasta格式的氨基酸序列：（例：>p5 NEMSWWMSHLIA）', 
                    value='请输入...', 
                    height=2, 
                    max_chars=200, 
                    help='最大长度限制为200')
st.write('您的输入是', text)
if st.button('确定'):
    st.write('提交成功')
# text='>p5/nNEMSWWMSHLIA'
#定义函数
def read_fasta(fname):
    fname = fname.split(' ')
    fname = np.array(fname).reshape(1,2)
    seq_df = pd.DataFrame(data=fname, columns=["Id", "Sequence"])
    return seq_df

def del_data(inFile):
    seq = read_fasta(inFile)
    seqname = seq.to_numpy()
    newseq = []
    j = seqname.shape[0]
    for i in range(j):
        if 6 <= len(seqname[i][1])<=100:
            newseq.append(seqname[i])
    newseq = np.array(newseq)
    print('序列去除小于6序列后的维度：', newseq.shape)
    newseq = pd.DataFrame(data=newseq, columns=["Id", "Sequence"])
    return newseq

def pro_data(seq):
#    df_n = insert_PAAC(seq)
    df_n = insert_AAC(seq)
    df_n = insert_CKSAAGP(df_n)
    df_n = insert_CTD(df_n)
    # df_n = insert_DPC(df_n)
    # df_n = insert_GTPC(df_n)
    # df_n = insert_QSO(df_n)
    df_n = insert_AAE(df_n)
    df_n = insert_ASDC(df_n)
    return df_n

test= del_data(text)

def MRMD_ALL(X_train,y_train,X_v,y_v):
    clf = ExtraTreesClassifier()
    kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=10)
    avg_score_valid=[]
    scores_test = []
    X = X_train
    Y = y_train
    for train_index, test_index in kf.split(X,Y):
        kf_X_train,kf_X_test=X[train_index],X[test_index]
        kf_y_train,kf_y_test=Y[train_index],Y[test_index]
        clf.fit(kf_X_train,kf_y_train)
        scores_test.append(clf.score(X_v,y_v))
    scores = np.array(scores_test)
    avg_score_test = np.sum(scores)/10
    print("This is validation score: %s" % (avg_score_test))
    return avg_score_test


seq_X_train = pd.read_csv('2/X_train.csv')
seq_test = test
Seq_X_train = pro_data(seq_X_train)
Seq_test = pro_data(seq_test)
# Seq_test.to_csv('data/Seq_test_all.csv',index=False)
fea = Seq_test.columns[2:]
X_train = Seq_X_train[fea].to_numpy()
test = Seq_test[fea].to_numpy()

from sklearn.preprocessing import MinMaxScaler, StandardScaler 

#归一化
nms = MinMaxScaler()
X_train_norm = nms.fit_transform(X_train)
test_norm = nms.transform(test)
#标准化
stdsc = StandardScaler(with_mean=False)
X_train_std = stdsc.fit_transform(X_train)
test_std = stdsc.transform(test)
#归一化
test_norm1 = pd.DataFrame(test_norm)
# test_norm1.to_csv('data/test_norm.csv',index=False)

#fs_MRMD
test = test_norm1

X = pd.read_csv('2/Seq_X_train_all.csv')
feature_names = X.columns[2:]
X_train = pd.read_csv('2/X_train_norm.csv').to_numpy()
y_train = pd.read_csv('2/y_train.csv').to_numpy()
X_train1,X_v,y_train1,y_v = train_test_split(X_train,y_train,test_size=0.2,random_state=10)

X_V = pd.DataFrame(X_v)
y_V = pd.DataFrame(y_v)

from mRMD import run
mrmd_f = run('2/process_X2.csv')

num_index = []
for i in range(702):
    num_index.append(str(i))
    
num_index = np.array(num_index)

Test = pd.DataFrame(test)
Test.columns=num_index
test = Test[mrmd_f[:455]]
# test.to_csv('Feature_different/test_MRMD.csv',index = False)
#模型
X_train = pd.read_csv('2/X_train_MRMD.csv').to_numpy()
X_test = test
y_train = pd.read_csv('2/y_train_resampled.csv').to_numpy()
# y_test = pd.read_csv('2/data/test/y_test.csv').to_numpy()

import xgboost as xgb
model = xgb.XGBClassifier(random_state=10)
model.fit(X_train, y_train)

kf = KFold(n_splits=10, random_state=10,shuffle=True)
for train_index, test_index in kf.split(X_train, y_train):
    model.fit(X_train,y_train)
eval_dict = []
kf = KFold(n_splits=5, shuffle=True, random_state=10)#K-折叠交叉验证。
for train_index, test_index in kf.split(X_train, y_train):
    X_train_ = X_train[train_index]
    y_train_ = y_train[train_index]
    model.fit(X_train_,y_train_)
y_pred=model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
if y_pred[0]==1:
    st.success('提供的肽链是抗病毒肽')
else:
    st.error('提供的肽链不是抗病毒肽')
