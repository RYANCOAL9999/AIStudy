import re
import nltk
import warnings
import requests
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

from PIL import Image
from nltk import PorterStemmer
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from wordcloud import WordCloud,ImageColorGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

warnings.filterwarnings("ignore", category=DeprecationWarning)       # %matplotlib inline

def generalRequestRaw(url:str, **kwargs):
    return requests.get(url, **kwargs).raw

def generalAllWords(combine:any, key:str, label:str, tive:int):
    return ' '.join(text for text in combine[key][combine[label]==tive])

# extract hashtags from tweets
def Hashtags_Extract(x:any):
    hashtags=[]
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r'#(\w+)', i)
        hashtags.append(ht)
    return hashtags

# remove pattern with special text 
def remove_pattern(text:str, pattern:str):
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern, text)
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i, "", text)
    return text

def racistWord(
        requestsRaw:any,
        all_words:any, 
        background_color:str,
        height:int,
        width:int,
        figsizeHeight:int,
        figsizeWidth:int,
        interpolation:str
    ):
    # combining the image with the dataset
    Mask = np.array(Image.open(requestsRaw))
    # We use the ImageColorGenerator library from Wordcloud 
    # Here we take the color of the image and impose it over our wordcloud
    image_colors = ImageColorGenerator(Mask)
    # Now we use the WordCloud function from the wordcloud library 
    wc = WordCloud(background_color=background_color, height=height, width=width, mask=Mask).generate(all_words)
    # Size of the image generated
    plt.figure(figsize=(figsizeWidth, figsizeHeight))
    # Here we recolor the words from the dataset to the image's color
    # recolor just recolors the default colors to the image's blue color
    # interpolation is used to smooth the image generated 
    plt.imshow(wc.recolor(color_func=image_colors), interpolation=interpolation)
    plt.axis('off')
    plt.show()

def hashtagsTive(tive, columns, header, n, y, x):
    # unnest the list
    ht_tive_unnest = sum(tive, [])
    # Counting the frequency of the words having Sentiment with Postive or Negative
    word_freq_tive = nltk.FreqDist(ht_tive_unnest)
    print(word_freq_tive)
    # Creating a dataframe for the most frequently used words in hashtags
    df_tive = pd.DataFrame({'Hashtags':list(word_freq_tive.keys()), 'Count':list(word_freq_tive.values())})
    # print(df_tive.head(header))
    # Plotting the barplot for the 10 most frequent words used for hashtags
    df_tive_plot = df_tive.nlargest(n, columns=columns) 
    sns.barplot(data=df_tive_plot, y=y, x=x)
    print(sns.despine())


def extractFeatures(vectorizer, train, matrix, num, size, random):
    # Using the features from Bag-of-Words Model for training set
    df = pd.DataFrame(matrix.todense())
    print(df)
    # Using the features from Bag-of-Words Model for training set
    train_data = matrix[:num]
    print(train.todense())
    # Features
    return train_test_split(
        train_data, 
        train['label'], 
        test_size=size, 
        random_state=random
    )


train = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
train_original=train.copy()
# print(train.shape)
# print(train_original)

test = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv')
test_original=train.copy()
# print(test.shape)
# print(test_original)

combine = train.append(test, ignore_index=True, sort=True)
# print(combine.head())
# print(combine.tail())

# Removing Twitter Handles (@user)
combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")
# print(combine.head())

# Removing Punctuations, Numbers, and Special Characters
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
# print(combine.head(10))

# Removing Short Words
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# print(combine.head(10))

# Tokenization
tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())
# print(tokenized_tweet.head(10))

# Stemming
tokenized_tweet = tokenized_tweet.apply(lambda x: [PorterStemmer().stem(i) for i in x])
# print(tokenized_tweet.head(10))

# stitch these tokens back together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combine['Tidy_Tweets'] = tokenized_tweet
# print(combine.head())

racistWord(
    generalRequestRaw('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True),
    generalAllWords(combine, 'Tidy_Tweets', 'label', 0),        # Store all the words from the dataset which are non-racist/sexist
    'black',
    1500,
    4000,
    10,
    20,
    'hamming'
)

racistWord(
    generalRequestRaw('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True),
    generalAllWords(combine, 'Tidy_Tweets', 'label', 1),        # Store all the words from the dataset which are non-racist/sexist
    'black',
    1500,
    4000,
    10,
    20,
    'gaussian'
)

hashtagsTive(
    Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==0]), # A nested list of all the hashtags from the positive reviews from the dataset
    10,
    20,
    'Count',
    'Hashtags',
    'Count'
)

hashtagsTive(
    Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==1]), # A nested list of all the hashtags from the negative reviews from the dataset
    10,
    20,
    'Count',
    'Hashtags',
    'Count'
)

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow_matrix = bow_vectorizer.fit_transform(combine['Tidy_Tweets'])

# Bag-of-Words Features
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = extractFeatures(
    bow_vectorizer,
    train,
    bow_matrix,
    31962,
    0.3,
    2
)


tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

tfidf_matrix = tfidf.fit_transform(combine['Tidy_Tweets'])

# TF-IDF Features
x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = extractFeatures(
    tfidf,
    train,
    tfidf_matrix,
    31962,
    0.3,
    2
)

# Logistic Regression
Log_Reg = LogisticRegression(random_state=0, solver='lbfgs')
Log_Reg.fit(x_train_bow, y_train_bow)                                           # Fitting the Logistic Regression Model

# The first part of the list is predicting probabilities for label:0 
# and the second part of the list is predicting probabilities for label:1
prediction_bow = Log_Reg.predict_proba(x_valid_bow)
print(prediction_bow)

# Calculating the F1 score
# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
prediction_int = prediction_bow[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
print(prediction_int)
log_bow = f1_score(y_valid_bow, prediction_int)         # calculating f1 score
print(log_bow)

# Using TF-IDF Features
Log_Reg.fit(x_train_tfidf, y_train_tfidf)
prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)
print(prediction_tfidf)

# Calculating the F1 score
prediction_int = prediction_tfidf[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
print(prediction_int)
log_tfidf = f1_score(y_valid_tfidf, prediction_int)      # calculating f1 score
print(log_tfidf)

# Using Bag-of-Words Features
model_bow = XGBClassifier(random_state=22, learning_rate=0.9)
print(model_bow.fit(x_train_bow, y_train_bow))
xgb=model_bow.predict_proba(x_valid_bow)
print(xgb)

# Calculating the F1 score
# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb=xgb[:,1]>=0.3
# converting the results to integer type
xgb_int=xgb.astype(np.int)
xgb_bow=f1_score(y_valid_bow,xgb_int)                   # calculating f1 score
print(xgb_bow)

# Using TF-IDF Features
model_tfidf=XGBClassifier(random_state=29,learning_rate=0.7)
model_tfidf.fit(x_train_tfidf, y_train_tfidf)
# The first part of the list is predicting probabilities for label:0 
# and the second part of the list is predicting probabilities for label:1
xgb_tfidf=model_tfidf.predict_proba(x_valid_tfidf)
print(xgb_tfidf)

# Calculating the F1 score
# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb_tfidf=xgb_tfidf[:,1]>=0.3
# converting the results to integer type
xgb_int_tfidf=xgb_tfidf.astype(np.int)
score=f1_score(y_valid_tfidf, xgb_int_tfidf)            # calculating f1 score
print(score)

# Decision Tree
dct = DecisionTreeClassifier(criterion='entropy', random_state=1)
# Using Bag-of-Words Features
dct.fit(x_train_bow,y_train_bow)
dct_bow = dct.predict_proba(x_valid_bow)
print(dct_bow)

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_bow=dct_bow[:,1]>=0.3
# converting the results to integer type
dct_int_bow=dct_bow.astype(np.int)
# calculating f1 score
dct_score_bow=f1_score(y_valid_bow,dct_int_bow)
print(dct_score_bow)

# Using TF-IDF Features
dct.fit(x_train_tfidf, y_train_tfidf)
dct_tfidf = dct.predict_proba(x_valid_tfidf)
print(dct_tfidf)

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_tfidf=dct_tfidf[:,1]>=0.3
# converting the results to integer type
dct_int_tfidf=dct_tfidf.astype(np.int)
# calculating f1 score
dct_score_tfidf=f1_score(y_valid_tfidf,dct_int_tfidf)
print(dct_score_tfidf)


# Model Comparison
Algo=['LogisticRegression(Bag-of-Words)','XGBoost(Bag-of-Words)','DecisionTree(Bag-of-Words)','LogisticRegression(TF-IDF)','XGBoost(TF-IDF)','DecisionTree(TF-IDF)']
score = [log_bow,xgb_bow,dct_score_bow,log_tfidf,score,dct_score_tfidf]
compare=pd.DataFrame({'Model':Algo,'F1_Score':score},index=[i for i in range(1,7)])
# print(compare.T)

# show F1 Model Vs Score
plt.figure(figsize=(18, 5))
sns.pointplot(x='Model', y='F1_Score', data=compare)
plt.title('Model Vs Score')
plt.xlabel('MODEL')
plt.ylabel('SCORE')
plt.show()

# Using the best possible model to predict for the test data
test_tfidf = tfidf_matrix[31962:]
test_pred = Log_Reg.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('result.csv', index=False)

# Test dataset after prediction
res = pd.read_csv('result.csv')
print(res)

# Evaluation Metrics
sns.countplot(train_original['label'])
print(sns.despine())