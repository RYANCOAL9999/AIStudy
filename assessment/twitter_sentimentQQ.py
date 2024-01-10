import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

warnings.filterwarnings("ignore")

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def generateTextIloc(train):
    preprocessed_text = []

    for sentence in tqdm(train.values):
        sent = decontracted(sentence)
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text[0: 5]

stopwords= [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
    'won', "won't", 'wouldn', "wouldn't"
]

df = pd.read_csv('../input/training-data/train.csv')

# df.head()

df.drop(['selected_text'], axis = 1, inplace = True)

# df.head()

sns.countplot(x = 'sentiment', data = df)
plt.show()

# df.shape

df['text'].iloc[0]

len(df['text'].iloc[0])

# df['text'].head()

# df.info()

df['text'].iloc[0]

text_length_list = []
for i in range(len(df)):
    if isinstance(df['text'].iloc[i], float) == True:
        print(df['text'].iloc[i])

isinstance("suhas", float)

# df.info()

df.dropna(inplace = True)

# df.info()

df['text_length'] = df['text'].apply(lambda x: len(x))

# df.head()

df['text_words'] = df['text'].apply(lambda x: len(x.split()))

# df.head()

positive_df = df[df['sentiment'] == 'positive']
negative_df = df[df['sentiment'] == 'negative']
neutral_df = df[df['sentiment'] == 'neutral']

print("The shape of the dataframe that contains only the positive reviews is: {}".format(positive_df.shape))
print("The shape of the dataframe that contains only the negative reviews is: {}".format(negative_df.shape))
print("The shape of the dataframe that contains only the neutral reviews is:  {}".format(neutral_df.shape))

wordcloud = WordCloud(width = 500, height = 500)

# df.head()

# positive_df.head()

positive_text = []

for i in range(len(positive_df)):
    positive_text.append(positive_df['text'].iloc[i])
positive_text[:5]

wordcloud = WordCloud(stopwords = stopwords)
wordcloud.generate(''.join(positive_text))
plt.figure(figsize = (10, 10))
plt.imshow(wordcloud)
plt.show()

negative_text = []
for i in range(len(negative_df)):
    negative_text.append(negative_df['text'].iloc[i])
negative_text[0: 5]

wordcloud = WordCloud(stopwords = stopwords, background_color = 'white')
wordcloud.generate(''.join(negative_text))
plt.figure(figsize = (10, 10))
plt.imshow(wordcloud)
plt.show()

# df.head()

# negative_df.head()

# positive_df.head()

# df.head()

df.drop(['textID'], axis = 1, inplace = True)

# df.head()

x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.3, random_state = 50)

# x_train.shape

preprocessed_text_x_train = generateTextIloc(x_train)

for i in range(len(x_train)):
    x_train['text'].iloc[i] = preprocessed_text_x_train[i]

# x_cv.shape

preprocessed_text_x_cv = generateTextIloc(x_cv)

for i in range(len(x_cv)):
    x_cv['text'].iloc[i] = preprocessed_text_x_cv[i]

y_train_converted = LabelBinarizer().fit_transform(y_train)
y_cv_converted = LabelBinarizer().fit_transform(y_cv)

# y_cv_converted
x_train_text = TfidfVectorizer().fit_transform(x_train['text'])
x_cv_text = TfidfVectorizer().transform(x_cv['text'])

x_train_text.shape

model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (20619,)))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

to_categorical(np.arange(1, 10))

# y_train

y_train_converted = LabelEncoder().fit_transform(y_train)

# y_train_converted

y_train_final = to_categorical(y_train_converted)

# x_train.head()

x_train_dropped = x_train.drop(['text'], axis = 1)

# x_train.head()

# x_train['text'].head()

x_train_dropped = x_train.drop(['text'], axis = 1)

# x_train_dropped.head()

# x_cv.head()

x_cv_dropped = x_cv.drop(['text'], axis = 1)

# x_cv_dropped.head()

X_train_final = MinMaxScaler().fit_transform(x_train_dropped)
X_cv_final = MinMaxScaler().transform(x_cv_dropped)

X_train_final[0: 5]

X_cv_final[0: 5]

y_train_encoded = LabelEncoder().fit_transform(y_train)

y_cv_encoded = LabelEncoder().fit_transform(y_cv)

y_train_final = to_categorical(y_train_encoded)
y_cv_final = to_categorical(y_cv_encoded)

y_train_final[0: 5]

y_cv_final[0: 5]

y_cv_final[0: 5]

# x_train.head()

x_train_vectorized = CountVectorizer().fit_transform(x_train['text'])
x_cv_vectorized = CountVectorizer().transform(x_cv['text'])

# x_train_vectorized

X_train_final[0: 5]

x_train_bow_toarray = x_train_vectorized.toarray()
x_cv_bow_toarray = x_cv_vectorized.toarray()

x_train_new = np.concatenate((x_train_bow_toarray, X_train_final), axis = 1)
x_cv_new = np.concatenate((x_cv_bow_toarray, X_cv_final), axis = 1)

model = Sequential()
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

model.fit(x_train_new, y_train_final, epochs = 10, validation_data = (x_cv_new, y_cv_final))

accuracy = model.history.history['accuracy']
val_accuracy = model.history.history['val_accuracy']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs = np.arange(1, 11)
fig, ax = plt.subplots(1, 2, figsize = (20, 5))

sns.lineplot(x = epochs, y = accuracy, ax = ax[0])
sns.lineplot(x = epochs, y = val_accuracy, ax = ax[0])
ax[0].set_title('Accuracy Vs Epochs')
sns.lineplot(x = epochs, y = loss, ax = ax[1])
sns.lineplot(x = epochs, y = val_loss, ax = ax[1])
ax[1].set_title('Loss Vs Epochs')
plt.show()