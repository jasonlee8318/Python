#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import packages

import urllib.request
from konlpy.tag import Okt #pip install konlpy
from gensim.models.word2vec import Word2Vec #pip install gensim
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#영화 리뷰 데이터 받기

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")


# In[5]:


#데이터 읽어오기
train_data = pd.read_table('ratings.txt')
print(train_data)


# In[6]:


# NULL 값 존재 유무확인

print(train_data.isnull().values.any())


# In[7]:


train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인


# In[8]:


print(len(train_data)) # 전체 리뷰 개수 출력


# In[9]:


#document 열 확인
train_data['document']


# In[10]:


# 정규 표현식을 통한 한글 외 문자 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")


# In[11]:


#정제된 document 열 확인
train_data['document']


# In[12]:


# 형태소 분석기 OKT를 사용한 토큰화 작업
okt = Okt()
tokenized_data = []
for sentence in train_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    #temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    tokenized_data.append(temp_X)


# In[13]:


tokenized_data


# In[14]:


#word2vec 생성

from gensim.models import Word2Vec
model = Word2Vec(sentences = tokenized_data, size = 100, window = 5, min_count = 5, workers = 4, sg = 1)

#size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
#window = 컨텍스트 윈도우 크기
#min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
#workers = 학습을 위한 프로세스 수
#sg = 0은 CBOW, 1은 Skip-gram.


# In[15]:


# 완성된 임베딩 매트릭스의 크기 확인
model.wv.vectors.shape


# In[16]:


model.wv.most_similar("어벤져스")


# In[17]:


model.wv.most_similar("노잼")


# In[18]:


model.wv.most_similar("액션")

