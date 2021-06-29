# import Flask class from the flask module
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np
import pandas as pd
import regex as re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import pairwise_distances
from scipy import spatial
from xgboost import XGBClassifier


app = Flask(__name__)

# Load the model from the file
# recommendation_model = joblib.load('model/recommendation_model.pkl')
# user_final_rating = joblib.load('model/user_final_rating.pkl')

# print('Log: Loaded recommendation model')
sentiment_model = XGBClassifier()
sentiment_model.load_model('model/xgb_sentiment_model.json')
print('Log: Loaded sentiment model')
tfidf_vectorizer     = joblib.load('model/tfidf_vectorizer.pkl')
print('Log: Loaded tfidf vectorizer model')
items_pred = []
df = pd.read_csv('cleaned_data.csv')
print('Log: Loaded data')
data = pd.DataFrame()
data['reviews'] = df.reviews.astype('str')
# preproces reviews
# def remove_stopwords(text):
    
    # # Convert text to lowercase
    # text = text.lower()
    # # remove numbers 
    # text = re.sub(r'\d+', '', text)
    # # remove punctuation
    # text = "".join([char for char in text if char not in string.punctuation])
    # # Remove stopwords
    # stop_words = stopwords.words('english')
    # # tokenize into words
    # text = word_tokenize(text)
    # # remove stop words
    
    # stemmer= PorterStemmer()
    # text = [word for word in text if word not in stop_words]
    # Setmming
    # stems = [stemmer.stem(word) for word in text]
    # text = " ".join(stems)
    # return text
# def stemming(text):
#     stems = [stemmer.stem(word) for word in text]
#     text = " ".join(stems)
#     return text
# lowercase
# data.reviews = data.reviews.str.lower()
# print('Log: Preprocessed data(Lowercase)')

# # remove numbers, nextline 
# data.reviews = [re.sub(r'[\n\r\d]*','', str(x)) for x in data.reviews]
# print('Log: Preprocessed data(numbers)')
# # remove punctuations
# punctuations = string.punctuation
# data.reviews = data['reviews'].replace(punctuations,'')
# print('Log: Preprocessed data(Punctuations)')
# # Tokenize
# data.reviews = [word_tokenize(x) for x in data.reviews]
# print('Log: Preprocessed data(tokenized)')
# # remove stopwords
# stop_words = stopwords.words('english')
# data.reviews = data['reviews'].apply(lambda x: remove_stopwords(x))
# print('Log: Preprocessed data(Stopwords)')
# stemmer= PorterStemmer()
# data.reviews = data['reviews'].apply(lambda x: stemming(x))
# print('Log: Preprocessed data(stemming)')



# data.reviews = data.reviews.apply(lambda text: preprocess_text(text))
print('Log: Preprocessed data')

def vectorizer(reviews_df):
    reviews =  [review for review in reviews_df.reviews]
    print('Before Vectorized shape',reviews_df.shape)
    print('Before Vectorized shape',reviews_df)
    v = tfidf_vectorizer.transform(reviews)
    print('v')
    print(v)
    reviews_df = pd.DataFrame(v.toarray(), columns = tfidf_vectorizer.get_feature_names())
    print('After Vectorized shape',reviews_df.shape)
    reviews_df['name_'] = df['name']
    return reviews_df

data = vectorizer(data)
print('Log: Vectorized data')


ratings = pd.read_csv('sample30.csv')
print('Log: Building recommendation model')
ratings = ratings[['name','reviews_username','reviews_rating']]
ratings = ratings[~ratings.reviews_username.isna()]
ratings = ratings.groupby(by = ['name','reviews_username']).mean()
ratings.reset_index(inplace = True)
ratings.columns = ['item','user','rating']
df_pivot = ratings.pivot(index= 'user'
                        ,columns='item'
                        ,values='rating'
                        ).fillna(0)
print('Log: Built pivot table')                       
dummy_train = ratings.copy()                            
dummy_train = dummy_train.pivot(index='user'
                                ,columns='item'
                                ,values='rating'
                                ).fillna(1)
mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T
df_subtracted.fillna(0)
print('Log: Built recommendation model') 

def build_recommendation_model(username):
    print('Log: Building recommendation model')
    global df_subtracted
    user_correlation = []
    for i in df_subtracted.index:
      user_correlation.append( 1 - spatial.distance.cosine(df_subtracted.loc[i]
                                                                 ,df_subtracted.loc[username]))
    user_correlation = [i if i>0 else 0 for i in user_correlation ]
    print('Log: Built similarity matrix')
    user_correlation_df = pd.DataFrame(user_correlation,columns=['sim'])
    user_predicted_ratings = np.dot(user_correlation_df.T, df_pivot.fillna(0))    
    user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
    # user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    # user_correlation[np.isnan(user_correlation)] = 0
    # print('Log: Built similarity matrix') 
    # user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
    # user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
    # print('Log: Built recommendation model') 
    return user_final_rating
data['name_'] = df['name']
def get_top_items_from_sentiment_analysis(top_20):
    top_items = pd.DataFrame(data[data['name_'].isin(top_20)])
    print(top_items.head())
    top_5 = pd.DataFrame(columns=['item','score'])
    for i in top_20:
        item = top_items[top_items.name_==i]
        item.drop(columns = ['name_'],axis =1,inplace = True)
        print('------item------')
        score_array = sentiment_model.predict(item)
        score = sum(score_array)/len(score_array)
        print(i,score,len(score_array))
        top_5.loc[len(top_5)] = [i,score]
    top_5 = top_5.sort_values(by='score',ascending=False)
    print(top_5)
    return top_5['item'].values
# data['name_'] = df['name']
# def get_top_items_from_sentiment_analysis(top_20):
#     print('len(top20)',len(top_20))
#     global data
#     local_data = pd.DataFrame(data[data['name_'].isin(top_20)])
#     name_df = pd.DataFrame(local_data['name_'],columns=['name_'])
#     print('local_data shape',local_data.shape)
#     # local_data = data[data['name_'].isin(top_20)]
#     print('local_data shape',local_data.shape)
#     print(local_data.name_.value_counts())
#     local_data.reviews = local_data.reviews.apply(lambda text: preprocess_text(text))
#     print('local_data shape',local_data.shape)
#     print('Log: Preprocessed data')
#     print('sum(local_data.name_.value_counts()) : ',sum(local_data.name_.value_counts()))
#     top_items = vectorizer(local_data,top_20)
#     print('data shape',data.shape)
#     print('top_items shape',top_items.shape)
#     print('Log: Vectorized data')
#     top_items['name_'] = name_df.iloc[:,0]
#     print('sum(name_df.name_.value_counts()) : ',sum(name_df.name_.value_counts()))
#     print('Sum(top_items.name_.value_counts()) : ',sum(top_items.name_.value_counts()))
    
#     print('top_items shape',top_items.shape)
#     print(top_items.name_.value_counts())
#     print(top_items.head())
#     top_5 = pd.DataFrame(columns=['item','score'])
#     score_array = []
#     for i in top_20:
#         item = top_items[top_items.name_==i]
#         item.drop(columns = ['name_'],axis =1,inplace = True)
#         print('------item------ \n',i)
#         score_array = sentiment_model.predict(item)
#         print('score array', score_array)
#         score = sum(score_array)/len(score_array)
#         print(i,score,len(score_array))
#         top_5.loc[len(top_5)] = [i,score]
#     top_5 = top_5.sort_values(by='score',ascending=False)
#     print(top_5)
#     return top_5['item'].values

def get_top_items_from_recommendation_analysis(username):
    user_final_rating = build_recommendation_model(username)
    op = user_final_rating.loc[username].sort_values(ascending=False)[0:20]
    for i,j in zip(op.index.values,op):
        print(str(i)+' :',op[i])
    return list(op.index.values)

@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    user = request.form['user']
    if not user or user not in list(ratings.user):
        return render_template('index.html', items_pred = [],showPred= 'N', showError = 'Y')
    
    # output = recommendation_model.loc[user].sort_values(ascending=False)[0:20]
    top_20 = get_top_items_from_recommendation_analysis(user)
    # print(top_20)
    top5 = list(get_top_items_from_sentiment_analysis(top_20))

    return render_template('index.html', items_pred = top5,showPred= 'Y', showError = 'N', user = user.title())


@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        return predict()
    else:
        return render_template('index.html',items_pred = [],showPred='N', showError='N')

@app.route('/favicon.ico', methods=['GET'])
def index2():
    # return render_template('index.html',items_pred = [],showPred='N', showError='N')
    return send_from_directory("static", 'amazon.png')


if __name__ == '__main__':
    app.debug = True
    app.run()

@app.route("/static/<path:path>", methods=['GET'])
def static_dir(path):
    return send_from_directory("static", path)