# Clickbait-detection-using-Long-Short-term-Memory-

**Loading packages**

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn import preprocessing
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, GlobalMaxPooling1D, Embedding, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

**Mouting Drive and Loading the Dataset**

from google.colab import drive
drive.mount("/content/drive")
clickbait_data=pd.read_csv("/content/drive/MyDrive/click bait/dataset/clickbait_data.csv")
clickbait_data.head()

**Converting the text into Lowercase**

_The data is converted into lowercase to avoid ambiguity between same words in different cases like 'NLP', 'nlp' or 'Nlp'_
clickbait_data['headline']=clickbait_data['headline'].str.lower()

**Removing Punctuations**

_The punctuations are removed to increase the efficiency of the model. They are irrelevant because they provide no added information._

import string 
clickbait_data['headline']=clickbait_data['headline'].str.replace('[{}]'.format(string.punctuation), '')
clickbait_data.head()

**Removing Numbers**

clickbait_data['headline'] = clickbait_data['headline'].str.rstrip(string.digits)


**Removing Stopwords**

_The entire input that we started with has now been converted into a sequence of words that are no more related to each other. The next step is to remove stopwords. Stopwords are those words in a language that are used to frame a sentence but hold no specific meaning. For example, the English language has stop words like ‘a’, ‘this’, ‘of’, etc. So, here we scan each word in the input and check if it belongs to the set of stopwords that are present in the English language. This set of stopwords is already given to us by the ‘stopword’ library, the user need not explicitly define them._

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
clickbait_data['headline'] = clickbait_data['headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
clickbait_data.head()

**Creating List words in each Headline**

h_d = []
c=0
for i in clickbait_data['headline']:
    h_d.append(i.split())
print(h_d[:2])

**Creating Word2vec Model**

_Word2vec is a technique for natural language processing published in 2013. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence._

w2vc_model = Word2Vec(h_d, size=200, workers=32, min_count=1, window=3)
print(w2vc_model)

**Tokenization Of Data**

_It is the process of breaking down into smaller units. Since our input is already an individual sentence and we need to represent each word uniquely, we perform word tokenization. Here, each word in a sentence is split and considered as a single unit._

token = Tokenizer(24646)
token.fit_on_texts(clickbait_data['headline'])
text = token.texts_to_sequences(clickbait_data['headline'])
print(text)
text = pad_sequences(text)
print(text.shape)

**Storing Class Labels Into a variable**

y = clickbait_data['clickbait'].values

**Splitting into Train and Test sets**

_The dataset is splitted into training and testing sets. The percentage of training data is 80% and testing data is 20%._

_**split the data into train test split**_

X_train, X_test, y_train, y_test = train_test_split(np.array(text), y, test_size=0.2,stratify=y)

**Building The Model**

_**build the model**_

model = Sequential()
model.add(w2vc_model.wv.get_keras_embedding(True))
model.add(Dropout(0.2)) #to reduce overfitting 
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

**compile and train model**

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), epochs=3)

**Confusion Matrix**

pred = [round(i[0]) for i in model.predict(X_test)]
cm = confusion_matrix(y_test, pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)
plt.yticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)
plt.show()

from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
print("F1 score of the model")
print(f1_score(y_test,pred))
print("Accuracy of the model")
print(accuracy_score(y_test,pred))
print("Accuracy of the model in percentage")
print(accuracy_score(y_test,pred)*100,"%")
print("precision score of the model")
print(precision_score(y_test,pred))
print("recall score of the model")
print(recall_score(y_test,pred))

**Evaluationg the Model with Custom input**

test = ["It Is Hell There: Ukraine's Zelensky Says Russia Has Destroyed Donbas"]
token_text = pad_sequences(token.texts_to_sequences(test))
preds = [round(i[0]) for i in model.predict(token_text)]
for (text, pred) in zip(test, preds):
    label = 'Clickbait' if pred == 1.0 else 'Not Clickbait'
    print("{} - {}".format(text, label))
