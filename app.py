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
from xgboost import XGBClassifier


app = Flask(__name__)

# Load the model from the file
# recommendation_model = joblib.load('model/recommendation_model.pkl')
user_final_rating = joblib.load('model/user_final_rating.pkl')

print('Log: Loaded recommendation model')
sentiment_model = XGBClassifier()
sentiment_model.load_model('model/xgb_sentiment_model.json')
print('Log: Loaded sentiment model')
tfidf_vectorizer     = joblib.load('model/tfidf_vectorizer.pkl')
print('Log: Loaded tfidf vectorizer model')
items_pred = []
df = pd.read_csv('sample30.csv')
print('Log: Loaded data')
data = pd.DataFrame()
data['reviews'] = (df.reviews_title +' '+ df.reviews_text).astype('str')
# preproces reviews
def preprocess_text(text):
    
    # Convert text to lowercase
    text = text.lower()
    # remove numbers 
    text = re.sub(r'\d+', '', text)
    # remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    stop_words = stopwords.words('english')
    # tokenize into words
    text = word_tokenize(text)
    # remove stop words
    text = [word for word in text if word not in stop_words]
    # Setmming
    stemmer= PorterStemmer()
    stems = [stemmer.stem(word) for word in text]
    text = " ".join(stems)
    return text   
data.reviews = data.reviews.apply(lambda text: preprocess_text(text))
print('Log: Preprocessed data')
def vectorizer(reviews_df):
    reviews =  [review for review in reviews_df.reviews]
    v = tfidf_vectorizer.transform(reviews)
    reviews_df = pd.DataFrame(v.toarray(), columns = tfidf_vectorizer.get_feature_names())
    reviews_df['name_'] = df['name']
    return reviews_df
data = vectorizer(data)
print('Log: Vectorized data')

# print('Log: Building recommendation model')
# ratings = pd.read_csv('sample30.csv')
# ratings = ratings[['name','reviews_username','reviews_rating']]
# ratings = ratings[~ratings.reviews_username.isna()]
# ratings = ratings.groupby(by = ['name','reviews_username']).mean()
# ratings.reset_index(inplace = True)
# ratings.columns = ['item','user','rating']
# df_pivot = ratings.pivot(index= 'user'
#                         ,columns='item'
#                         ,values='rating'
#                         ).fillna(0)
# print('Log: Building pivot model')                        
# dummy_train = ratings.copy()                            
# dummy_train = dummy_train.pivot(index='user'
#                                 ,columns='item'
#                                 ,values='rating'
#                                 ).fillna(1)
# mean = np.nanmean(df_pivot, axis=1)
# df_subtracted = (df_pivot.T-mean).T
# user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
# user_correlation[np.isnan(user_correlation)] = 0
# print('Log: Built similarity matrix') 
# user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
# user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
# print('Log: Built recommendation model') 

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

def get_top_items_from_recommendation_analysis(username):
    op = user_final_rating.loc[username].sort_values(ascending=False)[0:10]
    for i,j in zip(op.index.values,op):
        print(str(i)+' :',op[i])
    return list(op.index.values)

@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    user = request.form['user']
    if not user or user not in list(user_final_rating.index):
        return render_template('index2.html', items_pred = [],showPred= 'N', showError = 'Y')
    
    # output = recommendation_model.loc[user].sort_values(ascending=False)[0:20]
    top_20 = get_top_items_from_recommendation_analysis(user)
    # print(top_20)
    top5 = list(get_top_items_from_sentiment_analysis(top_20))

    return render_template('index2.html', items_pred = top5,showPred= 'Y', showError = 'N', user = user.title())


@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        return predict()
    else:
        return render_template('index2.html',items_pred = [],showPred='N', showError='N')

if __name__ == '__main__':
    app.debug = True
    app.run()

@app.route("/static/<path:path>", methods=['GET'])
def static_dir(path):
    return send_from_directory("static", path)