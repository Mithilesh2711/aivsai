from flask import Flask, redirect, url_for, jsonify

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import urllib.parse as p
import re
import os
import pickle

import numpy as np
import pandas as pd
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

port = int(os.environ.get('PORT', 5000))

@app.route('/<stri>')
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
        print(video_response)
        items = video_response.get("items")[0]
        print("/////////////////////////////////////////////////////////////////")
        print(items)
        print("/////////////////////////////////////////////////////////////////")
        # get the snippet, statistics & content details from the video response
        snippet = items["snippet"]
        # get infos from the snippet
        channel_title = snippet["channelTitle"]
        title = snippet["title"]
        description = snippet["description"]

        if (len(description) > 1000):
            description = description[0:1000]

        print(f"""\
        Title: {title}
        Description: {description}
        Channel Title: {channel_title}
        """)

        USvids = pd.read_csv("USvideos.csv", header=0)
        USvids.head(3)

        keep_columns = ['title', 'category_id']
        new_USvids = USvids[keep_columns]
        new_USvids.to_csv("newUS.csv", index=False)
        new_USvids = pd.read_csv("newUS.csv", header=0, names=['Title', 'Category_ID'])

        Categories_JSON = pd.read_json("US_category_id.JSON")
        Categories_JSON.head(3)

        CategoryDict = [{'id': item['id'], 'title': item['snippet']['title']} for item in Categories_JSON['items']]

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

        return FinalDF

    video_url = "https://www.youtube.com/watch?v=" + stri
    # parse video ID from URL
    video_id = get_video_id_by_url(video_url)
    # make API call to get video info
    response = get_video_details(youtube, id=video_id)
    # print extracted video infos
    final = print_video_infos(response)
    final_list = final.values.tolist()

    return jsonify(
        category = final_list[0][0],
    )

    #############################################################################


if(__name__ == "__main__"):
    app.run(host='0.0.0.0', port=port)