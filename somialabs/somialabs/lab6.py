# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import matplotlib
import textmining as tm
from ipywidgets import (
    interact,
    interactive,
    fixed,
    interact_manual,
    widgets
)
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
from nltk.corpus import stopwords
from itertools import chain
import string
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import sys, time
import pyLDAvis
import pyLDAvis.gensim
from gensim.models import Phrases
from gensim import corpora
from gensim import models
import numpy as np

remove_stopwords = lambda x: ' '.join(y for y in x.split() if y not in tm.stopwords)
stem_words = lambda x: ' '.join(tm.stem(x.split()))
def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return ''
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join(list(map(lambda list_tokens_POS: [
            [
                lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) 
                if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
            ] 
            for tokens_POS in list_tokens_POS
        ], [[pos_tag(tokens) for tokens in [word_tokenize(text)]]]))[0][0])

def load_imdb_data():
    imdb_data = pd.read_csv("imdb_labelled.txt", header=None, sep="\t", names=["review", "sentiment"])
    return imdb_data

def preprocess_imdb_data(
    remove_punctuaton = True,
    lower = True,
    remove_http = True,
    remove_at = True,
    remove_short = True,
    remove_stop = True,
    remove_more_stop = True,
    stem = False,
    lemmatize_words = True):
    imdb_data = load_imdb_data()
    if remove_punctuaton:
        imdb_data.review = imdb_data.review.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    if lower:
        imdb_data.review = imdb_data.review.str.lower()
    if remove_http:
        imdb_data.review = imdb_data.review.str.replace(r'http\S+', '')
    if remove_at:
        imdb_data.review = imdb_data.review.str.replace(r'@\S+', '')
    if remove_short:
        imdb_data.review = imdb_data.review.str.findall(r'\w{3,}').str.join(' ')
    if remove_stop:
        imdb_data.review = imdb_data.review.apply(remove_stopwords)
    if remove_more_stop:
        extra_stop_words = remove_more_stop.split(',')
        remove_more_stop_words = lambda x: ' '.join(y for y in x.split() if y not in extra_stop_words)
        imdb_data.review = imdb_data.review.apply(remove_more_stop_words)
    if stem:
        imdb_data.review = imdb_data.review.map(stem_words)
    if lemmatize_words:
        imdb_data.review = imdb_data.review.map(lemmatize)
    return imdb_data

def interact_imdb_data():
    interact(preprocess_imdb_data,
        remove_punctuaton=widgets.Checkbox(value=False, continuous_update=False),
        lower=widgets.Checkbox(value=False, continuous_update=False),
        remove_http=widgets.Checkbox(value=False, continuous_update=False),
        remove_at=widgets.Checkbox(value=False, continuous_update=False),
        remove_short=widgets.Checkbox(value=False, continuous_update=False),
        remove_stop=widgets.Checkbox(value=False, continuous_update=False),
        remove_more_stop=widgets.Text(value='aar,år', description="More stopwords:", continuous_update=False),
        stem=widgets.Checkbox(value=False, continuous_update=False),
        lemmatize_words=widgets.Checkbox(value=False, continuous_update=False),
    )
    
def word_frequency_plot(
    remove_punctuaton = True,
    lower = True,
    remove_http = True,
    remove_at = True,
    remove_short = True,
    remove_stop = True,
    remove_more_stop = True,
    stem = False,
    lemmatize_words = True):
    imdb_data = preprocess_imdb_data(remove_punctuaton, lower, remove_http, remove_at, remove_short, remove_stop,
                                    remove_more_stop, stem, lemmatize_words)
    corpus = imdb_data.review
    count_vectorizer = CountVectorizer(min_df=1)
    term_frequencies_matrix = count_vectorizer.fit_transform(corpus)
    tdm_matrix = pd.DataFrame(data=term_frequencies_matrix.toarray(), columns=count_vectorizer.get_feature_names())
    term_frequencies = tdm_matrix[[x for x in tdm_matrix.columns if len(x) > 1]].sum()
    term_frequencies.sort_values(ascending=False)[:30].plot.bar()

def interact_word_frequency_plot():
    interact(word_frequency_plot,
        remove_punctuaton=widgets.Checkbox(value=False, continuous_update=False),
        lower=widgets.Checkbox(value=False, continuous_update=False),
        remove_http=widgets.Checkbox(value=False, continuous_update=False),
        remove_at=widgets.Checkbox(value=False, continuous_update=False),
        remove_short=widgets.Checkbox(value=False, continuous_update=False),
        remove_stop=widgets.Checkbox(value=False, continuous_update=False),
        remove_more_stop=widgets.Text(value='aar,år', description="More stopwords:", continuous_update=False),
        stem=widgets.Checkbox(value=False, continuous_update=False),
        lemmatize_words=widgets.Checkbox(value=False, continuous_update=False))
    
def classifier_cross_validation(corpus_size, 
    remove_punctuaton = True,
    lower = True,
    remove_http = True,
    remove_at = True,
    remove_short = True,
    remove_stop = True,
    remove_more_stop = [],
    stem = False,
    lemmatize_words = True):
    imdb_data = preprocess_imdb_data(remove_punctuaton, lower, remove_http, remove_at, remove_short, remove_stop,
                                remove_more_stop, stem, lemmatize_words)
    sampled_imdb = imdb_data.head(corpus_size)
    X_train, X_test, y_train, y_test = train_test_split(sampled_imdb.review,
                                                        sampled_imdb.sentiment, random_state=0)
    print("Our training data has {} rows".format(len(X_train)))
    print("Our test data has {} rows".format(len(X_test)))    
    count_vectorizer = CountVectorizer(stop_words=None)
    X_train_counts = count_vectorizer.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    model = LinearSVC()
    classifier = model.fit(X_train_tfidf, y_train)
    vec = count_vectorizer.transform(X_test)
    predictions = classifier.predict(vec)
    tdf_vectorizer = TfidfVectorizer(min_df=1)
    scores = cross_val_score(classifier, tdf_vectorizer.fit_transform(X_test), predictions, scoring='accuracy', cv=5)
    cross_val_scores = pd.DataFrame(scores)
    _ = cross_val_scores.boxplot().set_title("Cross validation scores trained on {} records".format(corpus_size))
    return cross_val_scores.describe()

def interact_classifier_cross_validation():
    interact(classifier_cross_validation,
        corpus_size=widgets.IntSlider(min=60, max=len(load_imdb_data()), value=1000, continuous_update=False),
         remove_punctuaton=widgets.Checkbox(value=False, continuous_update=False),
        lower=widgets.Checkbox(value=False, continuous_update=False),
        remove_http=widgets.Checkbox(value=False, continuous_update=False),
        remove_at=widgets.Checkbox(value=False, continuous_update=False),
        remove_short=widgets.Checkbox(value=False, continuous_update=False),
        remove_stop=widgets.Checkbox(value=False, continuous_update=False),
        remove_more_stop=widgets.Text(value='aar,år', description="More stopwords:", continuous_update=False),
        stem=widgets.Checkbox(value=False, continuous_update=False),
        lemmatize_words=widgets.Checkbox(value=False, continuous_update=False)
    )
    
def lda_model_topics(remove_punctuaton = True,
    lower = True,
    remove_http = True,
    remove_at = True,
    remove_short = True,
    remove_stop = True,
    remove_more_stop = [],
    stem = False,
    lemmatize_words = True):
    imdb_data = preprocess_imdb_data(remove_punctuaton, lower, remove_http, remove_at, remove_short, remove_stop,
                                remove_more_stop, stem, lemmatize_words)
    imdb_data['sentences'] = imdb_data.review.map(sent_tokenize)
    imdb_data['tokens_sentences'] = imdb_data['sentences'].map(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
    imdb_data['POS_tokens'] = imdb_data['tokens_sentences'].map(lambda tokens_sentences: [pos_tag(tokens) for tokens in tokens_sentences])
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''
    lemmatizer = WordNetLemmatizer()
    imdb_data['tokens_sentences_lemmatized'] = imdb_data.POS_tokens.progress_map(
        lambda list_tokens_POS: [
            [
                lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) 
                if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
            ] 
            for tokens_POS in list_tokens_POS
        ]
    )
    imdb_data['tokens'] = imdb_data['tokens_sentences_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))
    imdb_data['tokens'] = imdb_data['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() and len(token)>1])
    tokens = imdb_data['tokens'].tolist()
    bigram_model = Phrases(tokens)
    trigram_model = Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])

    dictionary_LDA = corpora.Dictionary(tokens)
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]
    np.random.seed(123456)
    num_topics = 20
    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                      id2word=dictionary_LDA, \
                                      passes=4, alpha=[0.01]*num_topics, \
                                      eta=[0.01]*len(dictionary_LDA.keys()))
    print("Wait while the visualization loads (might take 30s or so)...")    
    vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
    pyLDAvis.enable_notebook()
    return pyLDAvis.display(vis)

def interact_lda_model_topics():
    interact(lda_model_topics,      
        lower=widgets.Checkbox(value=False, continuous_update=False),
        remove_http=widgets.Checkbox(value=False, continuous_update=False),
        remove_at=widgets.Checkbox(value=False, continuous_update=False),
        remove_short=widgets.Checkbox(value=False, continuous_update=False),
        remove_stop=widgets.Checkbox(value=False, continuous_update=False),
        remove_more_stop=widgets.Text(value='aar,år', description="More stopwords:", continuous_update=False),
        stem=widgets.Checkbox(value=False, continuous_update=False),
        lemmatize_words=widgets.Checkbox(value=False, continuous_update=False)
    )
    
print("Social Media and Digital Methods Lab 6 initialized")
for letter in "...":
    sys.stdout.write(letter)
    time.sleep(0.5)
print("It's the final one...")
for letter in "...":
    sys.stdout.write(letter)
    time.sleep(0.5)
print(".....yaaaaaaaaay!")