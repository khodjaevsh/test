
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = pd.read_csv("sd QQ list.csv")

datax = pd.get_dummies(data.Answers)
datax['Questions'] = data.Questions



# In[2]:


from collections import Counter
Counter(data.Answers)


# In[3]:


from sklearn.utils import shuffle

X, y = shuffle(datax['Questions'].values, datax.iloc[:,:8].values, random_state = 1 )

# Sample data view

print(X[:10])
print(y[:10])


# ## Tokenize reviews

# ## Reviews into sequences using keras

# In[4]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

mxwords = 6
tk = Tokenizer(lower = True)
tk.fit_on_texts(X)
X_seq = tk.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=mxwords, padding='post')


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.2, random_state = 1)


# In[6]:


batch_size = 40
X_train1 = X_train[batch_size:]
y_train1 = y_train[batch_size:]

X_valid = X_train[:batch_size]
y_valid = y_train[:batch_size]

print(len(tk.word_counts.keys()))


# # t-SNE visualisation

# In[95]:


from gensim.models import word2vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nltk

STOP_WORDS = nltk.corpus.stopwords.words('english')

vec = []

for sentence in datax.Questions:
    v = sentence.lower().split(' ')
    for word in v:

        if word in STOP_WORDS:
            v.remove(word)

    vec.append(v)

vec


# In[101]:


model = word2vec.Word2Vec(vec, size=10, window=6, min_count=5, workers=4)
#model.wv['bike']
model.most_similar('bus')


# In[97]:


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model)


# # Set up and train the model

# In[74]:


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, GRU

vocabulary_size = len(tk.word_counts.keys())+1
max_words = mxwords


embedding_size = 180
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(Bidirectional(LSTM(400, return_sequences=True)))
model.add(Dropout(0.2))

model.add(LSTM(200))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[75]:


model.fit(X_train1, y_train1, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=20)


# In[76]:


scores = model.evaluate(X_test, y_test, verbose=0) 
print("Test accuracy:", scores[1])  


# # My question check

# In[61]:


my_q = ['''
What is the arrival time?
''']

mr = tk.texts_to_sequences(my_q)
mr = pad_sequences(mr, maxlen = mxwords)
test_pred = model.predict(mr, verbose=0)


col_name = [x for x in datax.iloc[:,:8].columns]
preds = [item for sublist in test_pred for item in sublist]
col = np.argmax(test_pred)

print('{} : {}'.format(datax.iloc[:,col].name,np.max(test_pred)))
print('\n')
print(pd.DataFrame(list(zip(col_name, preds)), columns = ['Question', 'Probability']).
      sort_values(by ='Probability', ascending = False))
      

