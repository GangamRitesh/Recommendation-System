# import libraties
import re
import string
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import pairwise_distances
import joblib


# Filtering out the warnings
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_colwidth=1000

# Reading data file from file
df = pd.read_csv('sample30.csv')

# Checking the shape of data frame
df.shape

# Checking the columns 
df.columns.values

# Checking columns decription 
df.info()

# Checking the distribution of numeric columns
df.describe()

# Checking top 2 rows of data frame
df.head(2)

# Data Cleaning

# Storing the missing data and missing data percentage in the missing_data dataframe
def missing_values(df):
    missing_data  = pd.DataFrame(columns=['Column'
                                          ,'Type'
                                          ,'Missing_count'
                                          ,'Missing_percentage'])
    for i in df.columns:
        null_count = df[i].isna().sum()
        null_perc  = null_count/len(df)*100
        if null_count != 0:
            df2={'Column':i
            ,'Type': df[i].dtype
            ,'Missing_count': null_count
            ,'Missing_percentage' : null_perc
            }
            missing_data = missing_data.append(df2,ignore_index=True)
    return missing_data.sort_values(by = 'Missing_percentage',ascending= False)

#checking the missing values in the data frame
missing_data = missing_values(df)
missing_data

# Handling Missing values
# Dropping columns whith high Null value percentage
df.drop(columns = ['reviews_userProvince','reviews_userCity','reviews_didPurchase'],axis =1,inplace = True)
df.shape

# Dropping the column reviews_date as the column can't be used for analysis
df.drop(columns = ['reviews_date'],axis =1,inplace = True)
df.shape

df = df[~df.user_sentiment.isna()]
df.shape

missing_data = missing_values(df)
missing_data

# Creating a dataframe with columns requied from analysing the sentiment of the reviews
df_sent = df[['user_sentiment']]
# df_sent['reviews'] = df['brand'] +' '+ df['categories'] +' '+df['reviews_title']+' '+ df['reviews_text'] 
df_sent['reviews'] = df['reviews_title']+' '+ df['reviews_text'] 
df_sent.head()

# Converting the type of reviews column to String type
df_sent['reviews'] = df_sent['reviews'].astype('str')

# Mapping the user_sentiment column to numeric values (0 if negative and 1 if positive)
df_sent.user_sentiment = df_sent.user_sentiment.apply(lambda x: 0 if x == 'Negative' else 1)

# Checking the dataframe
print(df_sent.head())

# Text processing
# Convert text to lowercase
df_sent.reviews = df_sent.reviews.str.lower()

# remove numbers, nextline 
df_sent.reviews = [re.sub(r'[\n\r\d]*','', str(x)) for x in df_sent.reviews]

# remove punctuations (replacing all the characters other than alphabets and space with '')
df_sent['reviews'] = df_sent['reviews'].str.replace('[^\w\s]','')

# Tokenize
df_sent.reviews = [word_tokenize(x) for x in df_sent.reviews]

# Remove stopwords
stop_words = stopwords.words('english')
def remove_stopwords(word_tokens):
    # remove stop words
    text_processed = [word for word in word_tokens if word not in stop_words]
    return text_processed
df_sent.reviews = df_sent.reviews.apply(lambda word_tokens: remove_stopwords(word_tokens))

# Setmming
stemmer= PorterStemmer()
def stemmer_func(word_tokens):
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems
df_sent.reviews = df_sent.reviews.apply(lambda word_tokens: stemmer_func(word_tokens))

# Remove Rare words
# Storing all the words and count in a dictionary
word_count = {}
def get_vocab(df):
  vocab = set()
  reviews = df.reviews
  for word_tokens in reviews:
    for token in word_tokens:
      vocab.add(token)
  return len(vocab)
def get_word_count(df):
  reviews = df.reviews
  for word_tokens in reviews:
    for token in word_tokens:
      word_count[token] = word_count.get(token, 0) + 1
  return len(word_count)

def remove_rare_words(word_tokens):
    # remove rare words (count < 20 )
    words = [word for word in word_tokens if word_count.get(word) > 20]
    return words
print(get_vocab(df_sent))
print(get_word_count(df_sent))
df_sent.reviews = df_sent.reviews.apply(lambda word_tokens: remove_rare_words(word_tokens))
print(get_vocab(df_sent))

# Remove frequent words
# Removing all the words in revies which occured more than 10000 times
def remove_frequent_words(word_tokens):
    # remove rare words
    words = [word for word in word_tokens if word_count[word] < 10000]
    return words

print(get_vocab(df_sent))
print(get_word_count(df_sent))
df_sent.reviews = df_sent.reviews.apply(lambda word_tokens: remove_rare_words(word_tokens))
print(get_vocab(df_sent))

df_sent.reviews = df_sent.reviews.apply(lambda text: ' '.join(text))

joblib.dump(df_sent, 'data/processed_data.pkl')

# Creating TF-IDF model using TfidfVectorizer function  
reviews =  [review for review in df_sent.reviews]
tfidf = TfidfVectorizer()
tfidf_array = tfidf.fit_transform(reviews)
tfidf_df = pd.DataFrame(tfidf_array.toarray(), columns = tfidf.get_feature_names())


# Model Building on tfidf dataframe
# Splitting the data into trrain and test datasets TF-IDF model data
X_train,X_test, y_train, y_test = train_test_split(tfidf_df,df_sent.user_sentiment, train_size=0.8, stratify = df_sent.user_sentiment, random_state=100 )

# Handling Class Imbalance
y_train.value_counts(normalize = True)*100

# Handling imbalance using SMOTE technique
smote= SMOTE()
X_train,y_train = smote.fit_resample(X_train,y_train)

print('TF-IDF model (before smote)       : ',Counter(y_train))

y_train = pd.DataFrame(y_train)
y_train.columns = ['user_sentiment']
y_train


xgb_tf = XGBClassifier(random_state=42
                       ,n_jobs=-1
                       ,verbose=10
                       ,max_depth=30
                       ,min_samples_leaf=100)

xgb_tf.fit(X_train, y_train)

# Predicting the train and test
y_train_pred = xgb_tf.predict(X_train)
y_test_pred = xgb_tf.predict(X_test.to_numpy())

# Function to print model accuracy
def model_performance(y_true, y_pred):
  accuracy = metrics.accuracy_score(y_true,y_pred)
  precision= metrics.precision_score(y_true,y_pred)
  recall   = metrics.recall_score(y_true,y_pred)
  f1_score = metrics.f1_score(y_true,y_pred)
  print('Accuracy score : %.2f'% accuracy)
  print('Precision      : %.2f'% precision)
  print('Recall         : %.2f'% recall)
  print('F1 score       : %.2f'% f1_score)
  
# Model performance on train data
print('Model performance on train data')
model_performance(y_train_pred,y_train)
# Model performance on test data
print('Model performance on test data')
model_performance(y_test_pred,y_test)

# Save the model as a json in a file 
xgb_tf.save_model('xgb_sentiment_model.pkl')

# Save the model as a pickle in a file 
joblib.dump(tfidf, 'tfidf_vectorizer.pkl') 

# Load the model from the file 
xgb_sentiment_model = XGBClassifier()
xgb_sentiment_model.load_model('xgb_sentiment_model.pkl')

# Load the model from the file 
tfidf_model = joblib.load('tfidf_vectorizer.pkl')
print('models loaded')


# Building Recommendation model
# Item- Item based 
# Reading data file from csv.
df = pd.read_csv('sample30.csv')

# Preparing Data from recommender system
# Filtering the columns required for building recommendation model
df= df[['name','reviews_username','reviews_rating']]

# Dropping rows with no username
ratings = df[~df.reviews_username.isna()]

ratings = ratings.groupby(by = ['name','reviews_username']).mean()

ratings.reset_index(inplace = True)
# Renaming the columns
ratings.columns = ['items','users','ratings']



# Model building
# Splitting data into train and test

train, test = train_test_split(ratings, test_size=0.20, random_state=100)

df_pivot = train.pivot(index= 'users'
                        ,columns='items'
                        ,values='ratings'
                        ).T
print('Log: Building pivot model')                        

# Creating dummy train & dummy test dataset
dummy_train = train.copy()                            
# The items not rated by user is marked as 1 for prediction. 
dummy_train['ratings'] = dummy_train['ratings'].apply(lambda x: 0 if x>=1 else 1)

dummy_train = dummy_train.pivot(index='users'
                                ,columns='items'
                                ,values='ratings'
                                ).fillna(1)
# Building User Similarity Matrix
# Using adjusted Cosine
# Normalising the rating of the movie for each user
mean = np.nanmean(df_pivot, axis=1)
# Substracting the mean from the ratings column
df_subtracted = (df_pivot.T-mean).T.fillna(0)
# # Building Item Similarity Matrix using cosine similarity
item_correlation = 1 - pairwise_distances(df_subtracted, metric='cosine')
# Replacing the null with 0
item_correlation[np.isnan(item_correlation)] = 0
# Replacing regative values with 0
item_correlation[item_correlation<0]=0
print(item_correlation)
# Building prediction dataframe 
item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
# Building user item predicted rating matrix
item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
print('built item_final_rating')

user_input = 'joshua'
# Recommending the Top 5 products to the user.
top_10_recommended = item_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
top_10_recommended

# Evaluation - Item Item
common =  test[test['items'].isin(train['items'])]
common_item_based_matrix = common.pivot_table(index='users', columns='items', values='ratings').T
item_correlation_df = pd.DataFrame(item_correlation)

item_correlation_df['items'] = df_subtracted.index
item_correlation_df.set_index('items',inplace=True)
item_correlation_df.head()
list_name = common['items'].tolist()
item_correlation_df.columns = df_subtracted.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]
item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T

item_correlation_df_3[item_correlation_df_3<0]=0

common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings

dummy_test = common.copy()

dummy_test['ratings'] = dummy_test['ratings'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='users', columns='items', values='ratings').T.fillna(0)

common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)


common_ = common.pivot_table(index='users', columns='items', values='ratings').T

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(common_)
print(y)
# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum(((common_ - y )**2).fillna(0).values))/total_non_nan)**0.5
print(rmse)