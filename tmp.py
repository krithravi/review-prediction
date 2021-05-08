# imports
import json
import numpy as np
import nltk
import spacy
import re
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import mord
from sklearn import linear_model, metrics, preprocessing

# constants should we want that

# taken from wikipedia + stackexchange answer: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "i'd": "i had / i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}

# a bit of set up
lemmatization_model = spacy.load('en_core_web_sm')

# make list of entries
def makeListEntries(filename):
    data = [json.loads(line) for line in open(filename, 'r')]

    for entry in data:
        entry['review_body'] = entry['review_body'].lower()
        entry['review_body'] = entry['review_body'].replace("’", "'")

        # taking out contractions
        for key in contractions:
            entry['review_body'] = re.sub(key, contractions[key], entry['review_body'])

        entry['tokenized'] = []

        # removing unnecessary punctuation
        tokens = lemmatization_model(entry['review_body'])
        entry['tokenized'] = [token.lemma_ for token in tokens if token.lemma_ not in {',', '.'}]
    return data

# vectorize

# makes the list of words a string, adds that to a list
def makeListText(dataSet):
    resList = []
    for entry in dataSet:
        resList.append(" ".join(entry['tokenized']))
    return resList

# deal with target (the stars) as well
def makeListStars(dataSet):
    resList = []
    for entry in dataSet:
        resList.append(int(entry['stars']))
    return resList

### vader thresholds for scaling (constants)
"""
vader brackets for scaling!

1: [q0, q1)
2: [q1, q2)
3: [q3, q4)
4: [q4, q5)
5: [q5, q6]
"""
q0 = -1
q1 = -0.6
q2 = -0.2
q3 = 0.2
q4 = 0.6
q5 = 1

# taken from https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c
class OrdinalClassifier():
    
    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}
    
    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf
    
    def predict_proba(self, X):
        clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i,y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                #predicted.append(1 - clfs_predict[y][:,1])
                predicted.append(1 - clfs_predict[y-1][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                #predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
                predicted.append(clfs_predict[y-2][:,1] - clfs_predict[y-1][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                #predicted.append(clfs_predict[y-1][:,1])
                predicted.append(clfs_predict[y-2][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def betterScoring(true, pred):
    ## SCORING GUIDELINES:
    # 5 pts if right
    # 4 pts if off by 1
    # 3 pts if off by 2, etc
    size = len(true)
    score = 0
    
    thisdict = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }
    
    
    for i in range(size):
        offBy = abs(true[i] - pred[i])
        thisdict[offBy] += 1
        score += 5 - offBy
    
    #return score/(5 * size)
    return thisdict


def doAll(trainFileName, testFileName):
    trainSet = makeListEntries(trainFileName)
    testSet = makeListEntries(testFileName)
    """**************************************"""
    # data
    listTrainText = makeListText(trainSet)
    listTestText = makeListText(testSet)

    # target
    listTrainStars = makeListStars(trainSet)
    listTestStars = makeListStars(testSet)
    """*************************************"""
    # could do CountVectorizer
    cv = CountVectorizer(stop_words = 'english')

    trainCVMatr = cv.fit_transform(listTrainText)
    testCVMatr = cv.transform(listTestText)

    # could do TfidfVectorizer
    # tv = TfidfVectorizer(stop_words = 'english')

    # trainTVMatr = cv.fit_transform(listTrainText)
    # testTVMatr = cv.transform(listTestText)
    """*************************************"""
    # using CountVectorizer
    LR_CV_model = LogisticRegression(multi_class = 'multinomial', max_iter=1000, class_weight='balanced')
    LR_CV_model.fit(trainCVMatr, listTrainStars)

    # get it to predict
    LR_CV_prediction = LR_CV_model.predict(testCVMatr)

    # get accuracy score
    LR_CV_score = metrics.accuracy_score(listTestStars, LR_CV_prediction)
    LR_CV_f1 = metrics.f1_score(listTestStars, LR_CV_prediction, average='micro')
    LR_CV_r2 = metrics.r2_score(listTestStars, LR_CV_prediction, multioutput='variance_weighted')
    LR_my = betterScoring(listTestStars, LR_CV_prediction)
    # this is the bit with the tfidf vectorizer
    # LR_TV_model = LogisticRegression(multi_class = 'multinomial', max_iter=1000)
    # LR_TV_model.fit(trainTVMatr, listTrainStars)

    # get it to predict
    # LR_TV_prediction = LR_TV_model.predict(testTVMatr)

    # get accuracy score
    # LR_TV_score = metrics.accuracy_score(listTestStars, LR_TV_prediction)

    # what do the data say?
    #print("Multiclass, logistic regression, CountVectorizer: " + str(LR_CV_score))
    #print("Multiclass, logistic regression, TfidfVectorizer: " + str(LR_TV_score))
    """*************************************"""
    # using CountVectorizer
    NB_CV_model = MultinomialNB()
    NB_CV_model.fit(trainCVMatr, listTrainStars)

    # get it to predict
    NB_CV_prediction = NB_CV_model.predict(testCVMatr)

    # get accuracy score
    NB_CV_score = metrics.accuracy_score(listTestStars, NB_CV_prediction)
    NB_CV_f1 = metrics.f1_score(listTestStars, NB_CV_prediction, average='micro')
    NB_CV_r2 = metrics.r2_score(listTestStars, NB_CV_prediction, multioutput='variance_weighted')
    NB_my = betterScoring(listTestStars, NB_CV_prediction)
    # this is the bit with the tfidf vectorizer
    # NB_TV_model = MultinomialNB()
    # NB_TV_model.fit(trainCVMatr, listTrainStars)

    # get it to predict
    # NB_TV_prediction = NB_TV_model.predict(testTVMatr)

    # get accuracy score
    # NB_TV_score = metrics.accuracy_score(listTestStars, NB_TV_prediction)

    # what do the data say?
    #print("Naive Bayes, CountVectorizer: " + str(NB_CV_score))
    # print("Naive Bayes, TfidfVectorizer: " + str(NB_TV_score))
    """*************************************"""
    sid = SentimentIntensityAnalyzer()
    listOfRes = []

    data2 = [json.loads(line) for line in open(testFileName, 'r')]

    for entry in data2:
        listOfRes.append(sid.polarity_scores(entry['review_body'])['compound'])

    scaledRes = []
    size = len(listOfRes)
    for i in range(size):
        num = listOfRes[i]
        score = -1
        if num >= q0 and num < q1:
            score = 1
        elif num >= q1 and num < q2:
            score = 2
        elif num >= q2 and num < q3:
            score = 3
        elif num >= q3 and num < q4:
            score = 4
        elif num >= q4 and num <= q5:
            score = 5

        # add score back in
        scaledRes.append(score)

    vader_acc = metrics.accuracy_score(listTestStars, scaledRes)
    vader_f1 = metrics.f1_score(listTestStars, scaledRes, average='micro')
    vader_r2 = metrics.r2_score(listTestStars, scaledRes, multioutput='variance_weighted')
    vader_my = betterScoring(listTestStars, scaledRes)
    """*************************************"""
    # dealing with the ordinal regression
    ord_model = OrdinalClassifier(DecisionTreeClassifier())
    ord_model.fit(trainCVMatr, listTrainStars)
    ord_model_prediction = ord_model.predict(testCVMatr)
    
    size = len(listTestStars)
    for i in range(size):
        if (ord_model_prediction[i] < 1):
            ord_model_prediction[i] = 1

    ord_acc = metrics.accuracy_score(listTestStars, ord_model_prediction)
    ord_f1 = metrics.f1_score(listTestStars, ord_model_prediction, average='micro')
    ord_r2 = metrics.r2_score(listTestStars, ord_model_prediction, multioutput='variance_weighted')
    ord_my = betterScoring(listTestStars, ord_model_prediction)
    """*************************************"""
    # trying mord
    
    arr = np.asarray(listTrainStars)
    clf2 = mord.LogisticAT(alpha=1.)
    clf2.fit(trainCVMatr, arr)
    clf2_prediction = clf2.predict(testCVMatr)
    
    LAT_acc = metrics.accuracy_score(listTestStars, clf2_prediction)
    LAT_f1 = metrics.f1_score(listTestStars, clf2_prediction, average='micro')
    LAT_r2 = metrics.r2_score(listTestStars, clf2_prediction, multioutput='variance_weighted')
    LAT_my = betterScoring(listTestStars, clf2_prediction)
    #print('AccuracyScore of LogisticAT %s' %
          #metrics.accuracy_score(listTestStars, clf2.predict(testCVMatr)))
    
    clf3 = mord.LogisticIT(alpha=1.)
    clf3.fit(trainCVMatr, arr)
    clf3_prediction = clf3.predict(testCVMatr)
    
    LIT_acc = metrics.accuracy_score(listTestStars, clf3_prediction)
    LIT_f1 = metrics.f1_score(listTestStars, clf3_prediction, average='micro')
    LIT_r2 = metrics.r2_score(listTestStars, clf3_prediction, multioutput='variance_weighted')
    LIT_my = betterScoring(listTestStars, clf3_prediction)
    #print('AccuracyScore of LogisticIT %s' %
          #metrics.accuracy_score(listTestStars, clf3.predict(testCVMatr)))

    clf4 = mord.LogisticSE(alpha=1.)
    clf4.fit(trainCVMatr, arr)
    clf4_prediction = clf4.predict(testCVMatr)
    
    LSE_acc = metrics.accuracy_score(listTestStars, clf4_prediction)
    LSE_f1 = metrics.f1_score(listTestStars, clf4_prediction, average='micro')
    LSE_r2 = metrics.r2_score(listTestStars, clf4_prediction, multioutput='variance_weighted')
    LSE_my = betterScoring(listTestStars, clf4_prediction)
    #print('AccuracyScore of LogisticSE %s' %
        #metrics.accuracy_score(listTestStars, clf4.predict(testCVMatr)))
    """*************************************"""
    

    # return value
    categoryName = trainFileName.replace("dataset/prodAnalysis/train_", "")
    categoryName = categoryName.replace(".json", "")
    return [categoryName, 
            LR_CV_score, LR_CV_f1, LR_CV_r2, LR_my, 
            NB_CV_score, NB_CV_f1, NB_CV_r2, NB_my, 
            vader_acc, vader_f1, vader_r2, vader_my, 
            ord_acc, ord_f1, ord_r2, ord_my,
            LAT_acc, LAT_f1, LAT_r2, LAT_my, 
            LIT_acc, LIT_f1, LIT_r2, LIT_my, 
            LSE_acc, LSE_f1, LSE_r2, LSE_my, 
    ]

# run 'em all
doAll("dataset/smol_train.json", "dataset/smol_test.json")
#doAll("dataset/dataset_en_train.json", "dataset/dataset_en_test.json")#

listSubFiles = [
    ["dataset/prodAnalysis/train_entire.json", "dataset/prodAnalysis/test_entire.json"],
    ["dataset/prodAnalysis/train_apparel.json", "dataset/prodAnalysis/test_apparel.json"],
    ["dataset/prodAnalysis/train_automotive.json", "dataset/prodAnalysis/test_automotive.json"],
    ["dataset/prodAnalysis/train_baby_product.json", "dataset/prodAnalysis/test_baby_product.json"],
    ["dataset/prodAnalysis/train_beauty.json", "dataset/prodAnalysis/test_beauty.json"],
    ["dataset/prodAnalysis/train_book.json", "dataset/prodAnalysis/test_book.json"],
    ["dataset/prodAnalysis/train_camera.json", "dataset/prodAnalysis/test_camera.json"],
    ["dataset/prodAnalysis/train_digital_ebook_purchase.json", "dataset/prodAnalysis/test_digital_ebook_purchase.json"],
    ["dataset/prodAnalysis/train_digital_video_download.json", "dataset/prodAnalysis/test_digital_video_download.json"],
    ["dataset/prodAnalysis/train_drugstore.json", "dataset/prodAnalysis/test_drugstore.json"],
    ["dataset/prodAnalysis/train_electronics.json", "dataset/prodAnalysis/test_electronics.json"],
    ["dataset/prodAnalysis/train_furniture.json", "dataset/prodAnalysis/test_furniture.json"],
    ["dataset/prodAnalysis/train_grocery.json", "dataset/prodAnalysis/test_grocery.json"],
    ["dataset/prodAnalysis/train_home.json", "dataset/prodAnalysis/test_home.json"],
    ["dataset/prodAnalysis/train_home_improvement.json", "dataset/prodAnalysis/test_home_improvement.json"],
    ["dataset/prodAnalysis/train_industrial_supplies.json", "dataset/prodAnalysis/test_industrial_supplies.json"],
    ["dataset/prodAnalysis/train_jewelry.json", "dataset/prodAnalysis/test_jewelry.json"],
    ["dataset/prodAnalysis/train_kitchen.json", "dataset/prodAnalysis/test_kitchen.json"],
    ["dataset/prodAnalysis/train_lawn_and_garden.json", "dataset/prodAnalysis/test_lawn_and_garden.json"],
    ["dataset/prodAnalysis/train_luggage.json", "dataset/prodAnalysis/test_luggage.json"],
    ["dataset/prodAnalysis/train_musical_instruments.json", "dataset/prodAnalysis/test_musical_instruments.json"],
    ["dataset/prodAnalysis/train_office_product.json", "dataset/prodAnalysis/test_office_product.json"],
    ["dataset/prodAnalysis/train_other.json", "dataset/prodAnalysis/test_other.json"],
    ["dataset/prodAnalysis/train_pc.json", "dataset/prodAnalysis/test_pc.json"],
    ["dataset/prodAnalysis/train_personal_care_appliances.json", "dataset/prodAnalysis/test_personal_care_appliances.json"],
    ["dataset/prodAnalysis/train_pet_products.json", "dataset/prodAnalysis/test_pet_products.json"],
    ["dataset/prodAnalysis/train_shoes.json", "dataset/prodAnalysis/test_shoes.json"],
    ["dataset/prodAnalysis/train_sports.json", "dataset/prodAnalysis/test_sports.json"],
    ["dataset/prodAnalysis/train_toy.json", "dataset/prodAnalysis/test_toy.json"],
    ["dataset/prodAnalysis/train_video_games.json", "dataset/prodAnalysis/test_video_games.json"],
    ["dataset/prodAnalysis/train_watch.json", "dataset/prodAnalysis/test_watch.json"],
    ["dataset/prodAnalysis/train_wireless.json", "dataset/prodAnalysis/test_wireless.json"]
]

"""
largeDf = pd.DataFrame()
for i in range(0,31):
    list = doAll(listSubFiles[i][0], listSubFiles[i][1])
    largeDf[list[0]] = list[1:]

largeDf.head()
largeDf.to_csv(path_or_buf="res.csv")

"""


print("done")
