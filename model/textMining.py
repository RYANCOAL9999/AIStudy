import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer 

df = pd.read_json(
    "../dataSet/News_Category_Dataset_v3.json", 
    lines=True
)

print(df.dtypes)

len(df)

df.sample(3)

#<matplotlib.axes._subplots.AxesSubplot at 0x1a695a80508>
df.date.hist(
    figsize=(
        12, 
        6
    ), 
    color='#86bf91'
)

len(
    set(
        df['category'].values
    )
)

cmap = matplotlib.cm.get_cmap('Spectral')

rgba = [
    cmap(i) for i in np.linspace(
        0, 
        1, 
        len(set(df['category'].values))
    )
]

#<matplotlib.axes._subplots.AxesSubplot at 0x1a6942753c8>
df['category'].value_counts().plot(
    kind='bar', 
    color=rgba
)

df_orig = df.copy()

df = df_orig[df['category'].isin(['CRIME', 'COMEDY'])]

print(df.shape)

df.head()

#(6864, 6)
df = df.loc[
    :, [
        'headline',
        'category'
    ]
]

#<matplotlib.axes._subplots.AxesSubplot at 0x1a695c76388>
df['category'].value_counts().plot(
    kind='bar',
    color =[
        'r',
        'b'
    ]
)

sample_doc = [
    "Hello I am a boy", 
    "Hello I am a student", 
    "My name is Jill"
]

cv=CountVectorizer(max_df=0.85) 
word_count_vector=cv.fit_transform(sample_doc) 
word_count_vector_arr = word_count_vector.toarray()
## Wrong example - pd.DataFrame(word_count_vector_arr,columns=cv.vocabulary_) 
pd.DataFrame(
    word_count_vector_arr,
    columns=sorted(
        cv.vocabulary_, 
        key=cv.vocabulary_.get
    )
)

#{'hello': 2, 'am': 0, 'boy': 1, 'student': 7, 'my': 5, 'name': 6, 'is': 3, 'jill': 4}
cv.vocabulary_

docs=df['headline'].tolist()
# create a vocabulary of words,
# ignore words that appear in 85% of documents,
# eliminate stop words
newCV=CountVectorizer(max_df=0.95)
word_count_vector=newCV.fit_transform(docs)
# ['there', 'were', 'mass', 'shootings', 'in', 'texas', 'last', 'week', 'but', 'only']
list(newCV.vocabulary_.keys())[:10]

df['category_is_crime'] = df['category'] == 'CRIME'

X_train, X_test, y_train, y_test = train_test_split(
    word_count_vector, 
    df['category_is_crime'], 
    test_size=0.2, 
    random_state=42
)

modelExpression = LogisticRegression() 
modelExpression.fit(X_train, y_train)

y_pred = modelExpression.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
#[[766  26] [ 40 541]]
print(cm)
acc=(
        cm[0, 0]
        +cm[1, 1]
    )/sum(
        sum(cm)
    )

# Accuracy of a simple linear model with CountVectorizer is .... 95.19%
print(
    'Accuracy of a simple linear model with CountVectorizer is .... {:.2f}%'
    .format(
        acc*100
    )
)


