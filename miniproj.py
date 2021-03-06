from flask import Flask, redirect, url_for, jsonify

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import urllib.parse as p
import re
import os
import pickle
import logging

import numpy as np
import pandas as pd
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask_cors import CORS

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
CORS(app)

port = int(os.environ.get('PORT', 5000))

@app.route('/<stri>', methods =['GET'])
def fun(stri):

    SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

    def youtube_authenticate():
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        api_service_name = "youtube"
        api_version = "v3"
        client_secrets_file = "credentials.json"
        creds = None
        # the file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first time
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                creds = pickle.load(token)
        # if there are no (valid) credentials availablle, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
                creds = flow.run_local_server(port=0)
            # save the credentials for the next run
            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)

        return build(api_service_name, api_version, credentials=creds)

    # authenticate to YouTube API
    youtube = youtube_authenticate()

    def get_video_id_by_url(url):
        """
        Return the Video ID from the video `url`
        """
        # split URL parts
        parsed_url = p.urlparse(url)
        # get the video ID by parsing the query of the URL
        video_id = p.parse_qs(parsed_url.query).get("v")
        if video_id:
            return video_id[0]
        else:
            raise Exception(f"Wasn't able to parse video URL: {url}")

    def get_video_details(youtube, **kwargs):
        return youtube.videos().list(
            part="snippet,contentDetails,statistics",
            **kwargs
        ).execute()

    def print_video_infos(video_response):
        #print(video_response)

        for video_result in video_response.get('items', []):
            global description
            global title
            global channel_title
            channel_title =  video_result['snippet']['channelTitle']
            title =  video_result['snippet']['title']
            description =  video_result['snippet']['description']

        #items = video_response.get("items").[0]
        #print("/////////////////////////////////////////////////////////////////")
        #print(items)
        #print("/////////////////////////////////////////////////////////////////")
        # get the snippet, statistics & content details from the video response
        #snippet = items["snippet"]
        # get infos from the snippet
        # channel_title = snippet["channelTitle"]
        #title = snippet["title"]
        #description = snippet["description"]

        #if (len(description) > 1000):
            #description = description[0:1000]
            
        if (len(title) > len(description))    :
            description = title


        USvids = pd.read_csv("USvideos.csv", header=0)
        USvids.head(3)
        
        new_USvids = USvids['category_id']
        new_USvids.to_csv("newUS.csv", index=False)
        corpus = pd.read_csv("corpus.csv", header=0)
        new_USvids1 = pd.read_csv("newUS.csv", header=0)
        new_USvids1 = pd.DataFrame(new_USvids1['category_id'].values)
        corpus1 = pd.DataFrame(corpus['col'].values)
        
        new_USvids2 = pd.merge(corpus1, new_USvids1, left_index=True, right_index=True)
        
        new_USvids2.to_csv("newUS1.csv", index=False)
        new_USvids = pd.read_csv("newUS1.csv", header=0, names=['Title', 'Category_ID'])
        new_USvids = new_USvids.replace(np.nan, '', regex=True)
        
        
        global Categories    
        global CategoryDict
        #CategoryDict = [{'id': item['id'], 'title': item['snippet']['title']} for item in Categories_JSON['items']]
        CategoryDict = [{'id': '1', 'title': 'Film & Animation'},
        {'id': '2', 'title': 'Autos & Vehicles'},
        {'id': '10', 'title': 'Music'},
        {'id': '15', 'title': 'Pets & Animals'},
        {'id': '17', 'title': 'Sports'},
        {'id': '18', 'title': 'Short Movies'},
        {'id': '19', 'title': 'Travel & Events'},
        {'id': '20', 'title': 'Gaming'},
        {'id': '21', 'title': 'Videoblogging'},
        {'id': '22', 'title': 'People & Blogs'},
        {'id': '23', 'title': 'Comedy'},
        {'id': '24', 'title': 'Entertainment'},
        {'id': '25', 'title': 'News & Politics'},
        {'id': '26', 'title': 'Howto & Style'},
        {'id': '27', 'title': 'Education'},
        {'id': '28', 'title': 'Science & Technology'},
        {'id': '29', 'title': 'Nonprofits & Activism'},
        {'id': '30', 'title': 'Movies'},
        {'id': '31', 'title': 'Anime/Animation'},
        {'id': '32', 'title': 'Action/Adventure'},
        {'id': '33', 'title': 'Classics'},
        {'id': '34', 'title': 'Comedy'},
        {'id': '35', 'title': 'Documentary'},
        {'id': '36', 'title': 'Drama'},
        {'id': '37', 'title': 'Family'},
        {'id': '38', 'title': 'Foreign'},
        {'id': '39', 'title': 'Horror'},
        {'id': '40', 'title': 'Sci-Fi/Fantasy'},
        {'id': '41', 'title': 'Thriller'},
        {'id': '42', 'title': 'Shorts'},
        {'id': '43', 'title': 'Shows'},
        {'id': '44', 'title': 'Trailers'}]
        CategoriesDF = pd.DataFrame(CategoryDict)
        Categories = CategoriesDF.rename(index=str, columns={"id": "Category_ID", "title": "Category"})
        Categories.head(3)
        
        vector = CountVectorizer()
        counts = vector.fit_transform(new_USvids['Title'].values)
        
        NB_Model = MultinomialNB()
        targets = new_USvids['Category_ID'].values
        NB_Model.fit(counts, targets)
        
        MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        
        X = counts
        y = targets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
        
        NBtest = MultinomialNB().fit(X_train, y_train)
        nb_predictions = NBtest.predict(X_test)
        acc_nb = NBtest.score(X_test, y_test)
        print('The Naive Bayes Algorithm scored an accuracy of', acc_nb)
        
        Titles = [
                 description
            ]
        
        Titles_counts = vector.transform(Titles)
        Predict = NB_Model.predict(Titles_counts)
        
        CategoryNamesList = []
        for Category_ID in Predict:
            MatchingCategories = [x for x in CategoryDict if x["id"] == str(Category_ID)]
            if MatchingCategories:
                CategoryNamesList.append(MatchingCategories[0]["title"])
        
        TitleDataFrame = []
        for i in range(0, len(Titles)):
            TitleToCategories = {'Title': Titles[i], 'Category': CategoryNamesList[i]}
            TitleDataFrame.append(TitleToCategories)
        
        PredictDF = pd.DataFrame(Predict)
        TitleDF = pd.DataFrame(TitleDataFrame)
        PreFinalDF = pd.concat([PredictDF, TitleDF], axis=1)
        PreFinalDF.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
        FinalDF = PreFinalDF.drop(['Categ_ID'], axis=1)
        cols = FinalDF.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        FinalDF = FinalDF[cols]
        Categories
        CategoryDict

        return (FinalDF)

    video_url = "https://www.youtube.com/watch?v=" + stri
    # parse video ID from URL
    video_id = get_video_id_by_url(video_url)
    # make API call to get video info
    response = get_video_details(youtube, id=video_id)
    # print extracted video infos
    final = print_video_infos(response)
    final_list = final.values.tolist()

    return jsonify(
        {"category" : final_list[0][0]}
    )

    #############################################################################


if(__name__ == "__main__"):
    app.run(host='0.0.0.0', port=port)