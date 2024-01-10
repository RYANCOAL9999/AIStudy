import re
import nltk
import gensim
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.svm import SVC
from wordcloud import WordCloud
from nltk.corpus import stopwords
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from gensim.models.doc2vec import LabeledSentence
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

def hashtag_extract(x:any):
  hashtags = []
  for i in x:
    ht = re.findall(r"#(\w+)", i)
    hashtags.append(ht)
  return hashtags

def add_label(twt:any):
  output = []
  for i, s in zip(twt.index, twt):
    output.append(LabeledSentence(s, ["tweet_" + str(i)]))
  return output

train = pd.read_csv('drive/My Drive/Projects/Twitter Sentiment/train_tweet.csv')
test = pd.read_csv('drive/My Drive/Projects/Twitter Sentiment/test_tweets.csv')

# print(train.shape)
# print(test.shape)

# train.head()
# test.head()

train.isnull().any()
test.isnull().any()

# checking out the negative comments from the train set
train[train['label'] == 0].head(10)

# checking out the postive comments from the train set 
train[train['label'] == 1].head(10)

train['label'].value_counts().plot.bar(color = 'pink', figsize = (6, 4))

# checking the distribution of tweets in the data
length_train = train['tweet'].str.len().plot.hist(color = 'pink', figsize = (6, 4))
length_test = test['tweet'].str.len().plot.hist(color = 'orange', figsize = (6, 4))

# adding a column to represent the length of the tweet

train['len'] = train['tweet'].str.len()
test['len'] = test['tweet'].str.len()

# train.head(10)

# train.groupby('label').describe()

train.groupby('len').mean()['label'].plot.hist(color = 'black', figsize = (6, 4),)
plt.title('variation of length')
plt.xlabel('Length')
plt.show()

words = CountVectorizer(stop_words = 'english').fit_transform(train.tweet)

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
plt.title("Most Frequently Occuring Words - Top 30")

wordcloud = WordCloud(background_color = 'white', width = 1000, height = 1000).generate_from_frequencies(dict(words_freq))

plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize = 22)

normal_words =' '.join([text for text in train['tweet'][train['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state = 0, max_font_size = 110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Neutral Words')
plt.show()

negative_words =' '.join([text for text in train['tweet'][train['label'] == 1]])

wordcloud = WordCloud(background_color = 'cyan', width=800, height=500, random_state = 0, max_font_size = 110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('The Negative Words')
plt.show()

# collecting the hashtags

# extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(train['tweet'][train['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(train['tweet'][train['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})

# selecting top 20 most frequent hashtags     
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()), 'Count': list(a.values())})

# selecting top 20 most frequent hashtags     
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

# tokenizing the words present in the training set
tokenized_tweet = train['tweet'].apply(lambda x: x.split()) 

# creating a word to vector model
model_w2v = gensim.models.Word2Vec(
  tokenized_tweet,
  size=200, # desired no. of features/independent variables 
  window=5, # context window size
  min_count=2,
  sg = 1, # 1 for skip-gram model
  hs = 0,
  negative = 10, # for negative sampling
  workers= 2, # no.of cores
  seed = 34
)

model_w2v.train(tokenized_tweet, total_examples= len(train['tweet']), epochs=20)

model_w2v.wv.most_similar(positive = "dinner")

model_w2v.wv.most_similar(positive = "cancer")

model_w2v.wv.most_similar(positive = "apple")

model_w2v.wv.most_similar(negative = "hate")

tqdm.pandas(desc="progress-bar")

# label all the tweets
labeled_tweets = add_label(tokenized_tweet)

labeled_tweets[:6]

# removing unwanted patterns from the data
nltk.download('stopwords')

train_corpus = []

for i in range(0, 31962):
  review = re.sub('[^a-zA-Z]', ' ', train['tweet'][i])
  review = review.lower()
  review = review.split()
  
  # stemming
  review = [PorterStemmer().stem(word) for word in review if not word in set(stopwords.words('english'))]
  
  # joining them back with space
  review = ' '.join(review)
  train_corpus.append(review)

test_corpus = []

for i in range(0, 17197):
  review = re.sub('[^a-zA-Z]', ' ', test['tweet'][i])
  review = review.lower()
  review = review.split()

  # stemming
  review = [PorterStemmer().stem(word) for word in review if not word in set(stopwords.words('english'))]

  # joining them back with space
  review = ' '.join(review)
  test_corpus.append(review)

# creating bag of words
x = CountVectorizer(max_features = 2500).fit_transform(train_corpus).toarray()
y = train.iloc[:, 1]
# print(x.shape)
# print(y.shape)

# creating bag of words
x_test = CountVectorizer(max_features = 2500).fit_transform(test_corpus).toarray()
# print(x_test.shape)

# splitting the training data into train and valid sets
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25, random_state = 42)

# print(x_train.shape)
# print(x_valid.shape)
# print(y_train.shape)
# print(y_valid.shape)

# standardization
x_train = StandardScaler().fit_transform(x_train)
x_valid = StandardScaler().transform(x_valid)
x_test = StandardScaler().transform(x_test)

rFClassifiermodel = RandomForestClassifier()
rFClassifiermodel.fit(x_train, y_train)

y_pred = rFClassifiermodel.predict(x_valid)

print("Training Accuracy :", rFClassifiermodel.score(x_train, y_train))
print("Validation Accuracy :", rFClassifiermodel.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("F1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)

lrmodel = LogisticRegression()
lrmodel.fit(x_train, y_train)

y_pred = lrmodel.predict(x_valid)

print("Training Accuracy :", lrmodel.score(x_train, y_train))
print("Validation Accuracy :", lrmodel.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)

dtcmodel = DecisionTreeClassifier()
dtcmodel.fit(x_train, y_train)

y_pred = dtcmodel.predict(x_valid)

print("Training Accuracy :", dtcmodel.score(x_train, y_train))
print("Validation Accuracy :", dtcmodel.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)

svcmodel = SVC()
svcmodel.fit(x_train, y_train)

y_pred = svcmodel.predict(x_valid)

print("Training Accuracy :", svcmodel.score(x_train, y_train))
print("Validation Accuracy :", svcmodel.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm_first = confusion_matrix(y_valid, y_pred)
print(cm_first)

xgbmodel = XGBClassifier()
xgbmodel.fit(x_train, y_train)

y_pred = xgbmodel.predict(x_valid)

print("Training Accuracy :", xgbmodel.score(x_train, y_train))
print("Validation Accuracy :", xgbmodel.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm_last = confusion_matrix(y_valid, y_pred)
print(cm_last)