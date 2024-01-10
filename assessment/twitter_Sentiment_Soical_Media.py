#!/usr/bin/env python
# coding: utf-8

# # Social Media Sentiment Analysis 
# 
# ##### by Deepak Das

# ![sen](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/07/performing-twitter-sentiment-analysis1.jpg)

# # Problem Statement
# #### Dataset containing several tweets with positive and negative sentiment associated with it
# - Cyber bullying and hate speech has been a menace for quite a long time,So our objective for this task is to detect speeches tweets associated with negative sentiments.From this dataset we classify a tweet as hate speech if it has racist or sexist tweets associated with it.
# 
# - So our task here is to classify racist and sexist tweets from other tweets and filter them out.

# ![tweet](http://www.fuelaccounting.ca/wp-content/uploads/2015/04/twitter_logo-580-90.jpg)

# # Dataset Description
# 
# - The data is in csv format.In computing, a comma-separated values (CSV) file stores tabular data (numbers and text) in plain text.Each line of the file is a data record. Each record consists of one or more fields, separated by commas. 
# - Formally, given a training sample of tweets and labels, where label ‘1’ denotes the tweet is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist,our objective is to predict the labels on the given test dataset.

# # Attribute Information
# 
# - id : The id associated with the tweets in the given dataset
# - tweets : The tweets collected from various sources and having either postive or negative sentiments associated with it
# - label : A tweet with label '0' is of positive sentiment while a tweet with label '1' is of negative sentiment

# ## Importing the necessary packages 

# In[1]:


import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Train dataset used for our analysis

# In[2]:


train = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')


# #### We make a copy of training data so that even if we have to make any changes in this dataset we would not lose the original dataset.

# In[3]:


train_original=train.copy()


# #### Here we see that there are a total of 31692 tweets in the training dataset

# In[4]:


train.shape


# In[5]:


train_original


# ## Test dataset used for our analysis

# In[6]:


test = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv')


# #### We make a copy of test data so that even if we have to make any changes in this dataset we would not lose the original dataset.

# In[7]:


test_original=test.copy()


# #### Here we see that there are a total of 17197 tweets in the test dataset

# In[8]:


test.shape


# In[9]:


test_original


# ### We combine Train and Test datasets for pre-processing stage

# In[10]:


combine = train.append(test,ignore_index=True,sort=True)


# In[11]:


combine.head()


# In[12]:


combine.tail()


# # Data Pre-Processing

# ![pre](https://www.electronicsmedia.info/wp-content/uploads/2017/12/Data-Preprocessing.jpg)

# ##  Removing Twitter Handles (@user)

# Given below is a user-defined function to remove unwanted text patterns from the tweets. It takes two arguments, one is the original string of text and the other is the pattern of text that we want to remove from the string. The function returns the same input string but without the given pattern. We will use this function to remove the pattern ‘@user’ from all the tweets in our data.
# 

# In[13]:


def remove_pattern(text,pattern):
    
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,text)
    
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)
    
    return text
        


# In[14]:


combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")

combine.head()


# ## Removing Punctuations, Numbers, and Special Characters

# Punctuations, numbers and special characters do not help much. It is better to remove them from the text just as we removed the twitter handles. Here we will replace everything except characters and hashtags with spaces.

# In[15]:


combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")


# In[16]:


combine.head(10)


# ## Removing Short Words
# 

# We have to be a little careful here in selecting the length of the words which we want to remove. So, I have decided to remove all the words having length 3 or less. For example, terms like “hmm”, “oh” are of very little use. It is better to get rid of them.

# In[17]:


combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

combine.head(10)


#  ## Tokenization

# Now we will tokenize all the cleaned tweets in our dataset. Tokens are individual terms or words, and tokenization is the process of splitting a string of text into tokens.

# In[18]:


tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())
tokenized_tweet.head()


# ## Stemming

# Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”

# In[19]:


from nltk import PorterStemmer

ps = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

tokenized_tweet.head()


# #### Now let’s stitch these tokens back together.

# In[20]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combine['Tidy_Tweets'] = tokenized_tweet
combine.head()


# # Visualization from Tweets
# 

# ![vis](https://previews.123rf.com/images/mindscanner/mindscanner1404/mindscanner140401428/27857012-word-cloud-with-nlp-related-tags.jpg)

# ## WordCloud

# ![wc](https://az158878.vo.msecnd.net/marketing/product/42949674199/f89c7e08-56ac-457d-ae87-b6165e51432f/screen2.jpg)

# ### A wordcloud is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes.

# #### Importing Packages necessary for generating a WordCloud

# In[21]:


from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests


#  #### Store all the words from the dataset which are non-racist/sexist

# In[22]:


all_words_positive = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==0])


# #### We can see most of the words are positive or neutral. With happy, smile, and love being the most frequent ones. Hence, most of the frequent words are compatible with the sentiment which is non racist/sexists tweets. 

# In[23]:


# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_positive)

# Size of the image generated 
plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()


# #### Store all the words from the dataset which are racist/sexist

# In[24]:


all_words_negative = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==1])


# #### As we can clearly see, most of the words have negative connotations. So, it seems we have a pretty good text data to work on.

# In[25]:


# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_negative)

# Size of the image generated 
plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="gaussian")

plt.axis('off')
plt.show()


# # Understanding the impact of Hashtags on tweets sentiment

# ![hash](https://www.socialtalent.com/wp-content/uploads/2015/07/Twitter-Logo-Hashtag.png)

# ### Function to extract hashtags from tweets

# In[26]:


def Hashtags_Extract(x):
    hashtags=[]
    
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r'#(\w+)',i)
        hashtags.append(ht)
    
    return hashtags


# #### A nested list of all the hashtags from the positive reviews from the dataset

# In[27]:


ht_positive = Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==0])


# #### Here we unnest the list 

# In[28]:


ht_positive_unnest = sum(ht_positive,[])


# #### A nested list of all the hashtags from the negative reviews from the dataset

# In[29]:


ht_negative = Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==1])


# #### Here we unnest the list

# In[30]:


ht_negative_unnest = sum(ht_negative,[])


# ## Plotting BarPlots

# ![plot](https://www.mathworks.com/help/examples/graphics/win64/SingleDataSeriesExample_01.png)

# ### For Positive Tweets in the dataset

# #### Counting the frequency of the words having Positive Sentiment 

# In[31]:


word_freq_positive = nltk.FreqDist(ht_positive_unnest)

word_freq_positive


# #### Creating a dataframe for the most frequently used words in hashtags

# In[32]:


df_positive = pd.DataFrame({'Hashtags':list(word_freq_positive.keys()),'Count':list(word_freq_positive.values())})


# In[33]:


df_positive.head(10)


# #### Plotting the barplot for the 10 most frequent words used for hashtags 

# In[34]:


df_positive_plot = df_positive.nlargest(20,columns='Count') 


# In[35]:


sns.barplot(data=df_positive_plot,y='Hashtags',x='Count')
sns.despine()


# ### For Negative Tweets in the dataset

# #### Counting the frequency of the words having Negative Sentiment 

# In[36]:


word_freq_negative = nltk.FreqDist(ht_negative_unnest)


# In[37]:


word_freq_negative


# #### Creating a dataframe for the most frequently used words in hashtags

# In[38]:


df_negative = pd.DataFrame({'Hashtags':list(word_freq_negative.keys()),'Count':list(word_freq_negative.values())})


# In[39]:


df_negative.head(10)


# #### Plotting the barplot for the 10 most frequent words used for hashtags 

# In[40]:


df_negative_plot = df_negative.nlargest(20,columns='Count') 


# In[41]:


sns.barplot(data=df_negative_plot,y='Hashtags',x='Count')
sns.despine()


# # Extracting Features from cleaned Tweets

# ### Bag-of-Words Features

# Bag of Words is a method to extract features from text documents. These features can be used for training machine learning algorithms. It creates a vocabulary of all the unique words occurring in all the documents in the training set. 
# 
# Consider a corpus (a collection of texts) called C of D documents {d1,d2…..dD} and N unique tokens extracted out of the corpus C. The N tokens (words) will form a list, and the size of the bag-of-words matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens in document D(i).
# 
# For example, if you have 2 documents-
# 
# 
# 
# - D1: He is a lazy boy. She is also lazy.
# 
# - D2: Smith is a lazy person.
# 
# First, it creates a vocabulary using unique words from all the documents
# #### [‘He’ , ’She’ , ’lazy’ , 'boy’ ,  'Smith’  , ’person’] 
# 
# - Here, D=2, N=6
# 
# 
# 
# - The matrix M of size 2 X 6 will be represented as:
# 
# ![bow](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/07/table.png)
# 
# The above table depicts the training features containing term frequencies of each word in each document. This is called bag-of-words approach since the number of occurrence and not sequence or order of words matters in this approach.

# In[42]:


from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combine['Tidy_Tweets'])

df_bow = pd.DataFrame(bow.todense())

df_bow


# ### TF-IDF Features

# Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. 
# 
# Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
# 
# - TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization: 
# #### TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
# 
# - IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following: 
# #### IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
# 
# #### Example:
# 
# Consider a document containing 100 words wherein the word cat appears 3 times. The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.
# 
# ![tfidf](https://skymind.ai/images/wiki/tfidf.png)

# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')

tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'])

df_tfidf = pd.DataFrame(tfidf_matrix.todense())

df_tfidf


# # Applying Machine Learning Models

# ![ml](https://ak8.picdn.net/shutterstock/videos/23516428/thumb/12.jpg?i10c=img.resize(height:160))

# ### Using the features from Bag-of-Words Model for training set

# In[44]:


train_bow = bow[:31962]

train_bow.todense()


# ### Using features from TF-IDF for training set

# In[45]:


train_tfidf_matrix = tfidf_matrix[:31962]

train_tfidf_matrix.todense()


# ### Splitting the data into training and validation set

# In[46]:


from sklearn.model_selection import train_test_split


# #### Bag-of-Words Features

# In[47]:


x_train_bow,x_valid_bow,y_train_bow,y_valid_bow = train_test_split(train_bow,train['label'],test_size=0.3,random_state=2)


# #### Using TF-IDF features

# In[48]:


x_train_tfidf,x_valid_tfidf,y_train_tfidf,y_valid_tfidf = train_test_split(train_tfidf_matrix,train['label'],test_size=0.3,random_state=17)


# 
# ## Logistic Regression

# In[49]:


from sklearn.linear_model import LogisticRegression


# In[50]:


Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')


# ### Using Bag-of-Words Features 

# In[51]:


# Fitting the Logistic Regression Model

Log_Reg.fit(x_train_bow,y_train_bow)


# In[52]:


# The first part of the list is predicting probabilities for label:0 
# and the second part of the list is predicting probabilities for label:1
prediction_bow = Log_Reg.predict_proba(x_valid_bow)

prediction_bow


# #### Calculating the F1 score

# In[53]:


from sklearn.metrics import f1_score


# In[54]:


# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
prediction_int = prediction_bow[:,1]>=0.3

prediction_int = prediction_int.astype(np.int)
prediction_int

# calculating f1 score
log_bow = f1_score(y_valid_bow, prediction_int)

log_bow


# ### Using TF-IDF Features

# In[55]:


Log_Reg.fit(x_train_tfidf,y_train_tfidf)


# In[56]:


prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)

prediction_tfidf


# #### Calculating the F1 score

# In[57]:


prediction_int = prediction_tfidf[:,1]>=0.3

prediction_int = prediction_int.astype(np.int)
prediction_int

# calculating f1 score
log_tfidf = f1_score(y_valid_tfidf, prediction_int)

log_tfidf


# ## XGBoost

# In[58]:


from xgboost import XGBClassifier


# ### Using Bag-of-Words Features 

# In[59]:


model_bow = XGBClassifier(random_state=22,learning_rate=0.9)


# In[60]:


model_bow.fit(x_train_bow, y_train_bow)


# In[61]:


# The first part of the list is predicting probabilities for label:0 
# and the second part of the list is predicting probabilities for label:1
xgb=model_bow.predict_proba(x_valid_bow)

xgb


# #### Calculating the F1 score

# In[62]:


# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb=xgb[:,1]>=0.3

# converting the results to integer type
xgb_int=xgb.astype(np.int)

# calculating f1 score
xgb_bow=f1_score(y_valid_bow,xgb_int)

xgb_bow


# ### Using TF-IDF Features 

# In[63]:


model_tfidf=XGBClassifier(random_state=29,learning_rate=0.7)


# In[64]:


model_tfidf.fit(x_train_tfidf, y_train_tfidf)


# In[65]:


# The first part of the list is predicting probabilities for label:0 
# and the second part of the list is predicting probabilities for label:1
xgb_tfidf=model_tfidf.predict_proba(x_valid_tfidf)

xgb_tfidf


# #### Calculating the F1 score

# In[66]:


# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb_tfidf=xgb_tfidf[:,1]>=0.3

# converting the results to integer type
xgb_int_tfidf=xgb_tfidf.astype(np.int)

# calculating f1 score
score=f1_score(y_valid_tfidf,xgb_int_tfidf)

score


# ## Decision Tree

# In[67]:


from sklearn.tree import DecisionTreeClassifier


# In[68]:


dct = DecisionTreeClassifier(criterion='entropy', random_state=1)


# ### Using Bag-of-Words Features

# In[69]:


dct.fit(x_train_bow,y_train_bow)


# In[70]:


dct_bow = dct.predict_proba(x_valid_bow)

dct_bow


# In[71]:


# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_bow=dct_bow[:,1]>=0.3

# converting the results to integer type
dct_int_bow=dct_bow.astype(np.int)

# calculating f1 score
dct_score_bow=f1_score(y_valid_bow,dct_int_bow)

dct_score_bow


# ### Using TF-IDF Features

# In[72]:


dct.fit(x_train_tfidf,y_train_tfidf)


# In[73]:


dct_tfidf = dct.predict_proba(x_valid_tfidf)

dct_tfidf


# #### Calculating F1 Score

# In[74]:


# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_tfidf=dct_tfidf[:,1]>=0.3

# converting the results to integer type
dct_int_tfidf=dct_tfidf.astype(np.int)

# calculating f1 score
dct_score_tfidf=f1_score(y_valid_tfidf,dct_int_tfidf)

dct_score_tfidf


# # Model Comparison

# In[75]:


Algo=['LogisticRegression(Bag-of-Words)','XGBoost(Bag-of-Words)','DecisionTree(Bag-of-Words)','LogisticRegression(TF-IDF)','XGBoost(TF-IDF)','DecisionTree(TF-IDF)']


# In[76]:


score = [log_bow,xgb_bow,dct_score_bow,log_tfidf,score,dct_score_tfidf]

compare=pd.DataFrame({'Model':Algo,'F1_Score':score},index=[i for i in range(1,7)])


# In[77]:


compare.T


# In[78]:


plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='F1_Score',data=compare)

plt.title('Model Vs Score')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()


# ## Using the best possible model to predict for the test data
# 
# #### From the above comaprison graph we can see that Logistic Regression trained using TF-IDF features gives us the best performance

# In[79]:


test_tfidf = tfidf_matrix[31962:]


# In[80]:


test_pred = Log_Reg.predict_proba(test_tfidf)

test_pred_int = test_pred[:,1] >= 0.3

test_pred_int = test_pred_int.astype(np.int)

test['label'] = test_pred_int

submission = test[['id','label']]

submission.to_csv('result.csv', index=False)


# ### Test dataset after prediction

# In[81]:


res = pd.read_csv('result.csv')


# In[82]:


res


# # Summary
# 
# - From the given dataset we were able to predict on which class i.e Positive or Negative does the given tweet fall into.The following data was collected from Analytics Vidhya's site.
# 
# ### Pre-processing 
# 1. Removing Twitter Handles(@user)
# 2. Removing puntuation,numbers,special characters
# 3. Removing short words i.e. words with length<3
# 4. Tokenization
# 5. Stemming
# 
# ### Data Visualisation
# 1. Wordclouds
# 2. Barplots
# 
# ### Word Embeddings used to convert words to features for our Machine Learning Model
# 
# 1. Bag-of-Words 
# 2. TF-IDF 
# 
# ### Machine Learning Models used
# 1. Logistic Regression
# 2. XGBoost
# 3. Decision Trees 
# 
# ### Evaluation Metrics 
# - F1 score

# In[84]:


sns.countplot(train_original['label'])
sns.despine()


# ### Why use F1-Score instead of Accuracy ?
# 
# - From the above countplot generated above we see how imbalanced our dataset is.We can see that the values with label:0 i.e. positive sentiments are quite high in number as compared to the values with labels:1 i.e. negative sentiments.
# 
# 
# - So when we keep accuracy as our evaluation metric there may be cases where we may encounter high number of false positives.
# 
# #### Precison & Recall :- 
# - Precision means the percentage of your results which are relevant.
# - Recall refers to the percentage of total relevant results correctly classified by your algorithm
# ![met](https://cdn-images-1.medium.com/max/800/1*pOtBHai4jFd-ujaNXPilRg.png)
# 
# - We always face a trade-off situation between Precison and Recall i.e. High Precison gives low recall and vice versa.
# 
# 
# 
# 
# - In most problems, you could either give a higher priority to maximizing precision, or recall, depending upon the problem you are trying to solve. But in general, there is a simpler metric which takes into account both precision and recall, and therefore, you can aim to maximize this number to make your model better. This metric is known as F1-score, which is simply the harmonic mean of precision and recall.
# 
# ![f1](https://cdn-images-1.medium.com/max/800/1*DIhRgfwTcxnXJuKr2_cRvA.png)
# 
# 
# - So this metric seems much more easier and convenient to work with, as you only have to maximize one score, rather than balancing two separate scores.
