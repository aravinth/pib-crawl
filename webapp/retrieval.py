import time
import numpy as np
import langid
from ilmulti.segment import SimpleSegmenter, Segmenter
from ilmulti.sentencepiece import SentencePieceTokenizer
from datetime import timedelta 
import datetime
from webapp import db
from webapp.models import Entry, Link, Translation
from sqlalchemy import func
import itertools
from tqdm import tqdm
from ilmulti.translator.pretrained import mm_all
from bleualign.align import Aligner
import os
from ilmulti.utils.language_utils import inject_token
import csv
from sklearn.metrics.pairwise import cosine_similarity
from collections import namedtuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sqlalchemy import and_


def preprocess(corpus):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    processed = []
    corpus = corpus.splitlines()
    #corpus = sent_tokenize(corpus)
    for corp in corpus:
        translator = str.maketrans('','',string.punctuation)
        corp = corp.translate(translator)
        tokens = word_tokenize(corp)
        no_stop = [ps.stem(w.lower()) for w in tokens if not w in stop_words]
        alpha = [re.sub(r'\W+', '', a) for a in no_stop]
        alpha = [a for a in alpha if a]
        for a in alpha:
            processed.append(a)
    return ' '.join(processed)

def tfidf(query, candidates):
    vectorizer = TfidfVectorizer()
    candidate_features = vectorizer.fit_transform(candidates).toarray()
    query_feature = vectorizer.transform(query).toarray()
    N, d = candidate_features.shape
    # query_feature = np.tile(query_feature, (N, 1))
    similarities = cosine_similarity(query_feature, candidate_features)
    # similarities = np.diag(similarities)
    # indices = np.argsort(-1*similarities)
    # TODO(jerin): align indices with munkres
    # print(query_feature.shape, candidate_features.shape, similarities.shape)
    # print(similarities)
    indices = similarities.argmax(axis=1)
    similarities = similarities.max(axis=1)
    # return None, None
    return indices, similarities

def reorder(candidates, indices, similarities):
    Retrieved = namedtuple('Retrieved', 'id similarity')
    return [
        Retrieved(id=candidates[i].id, similarity=similarities[i]) \
        for i in indices
    ]

def get_candidates(query_id):
    delta = timedelta(days = 2)
    entry = db.session.query(Translation.parent_id, Entry.date)\
                        .filter(Translation.parent_id == Entry.id)\
                        .order_by(Entry.date.desc())\
                        .first()

    candidates = []
    matches = db.session.query(Entry.id) \
                .filter(Entry.lang=='en') \
                .filter(Entry.date.between(entry.date-delta,entry.date+delta))\
                .all()
    for match in matches:
        candidates.append(match.id)   
    return candidates

def bucket(query_id):
    # query id is always english
    delta = timedelta(days = 2)
    # All english samples near with a given time.
    entry = Entry.query.get(query_id)
    english = db.session.query(Entry) \
                .filter(Entry.lang=='en') \
                .filter(Entry.date.between(entry.date-delta,entry.date+delta))\
                .all()

    not_english = db.session.query(Entry.id) \
                .filter(Entry.lang!='en') \
                .filter(Entry.date.between(entry.date-delta,entry.date+delta))\
                .all()

    translations = []
    for candidate in not_english:
        translation = db.session.query(Translation)\
                            .filter(and_(Translation.parent_id == candidate.id,
                            Translation.lang == 'en'))\
                            .first()
        if translation is not None:
            # print(translation)
            translations.append(translation)

    # Translations and english can mismatch.
    # Cosine similarity works, therefore okay.
    return english, translations


def retrieve_neighbours2(query_id):
    candidates = get_candidates(query_id)
    corpus = []

    # Find all english samples near a time-delta

    query_content = Translation.query.filter(Translation.parent_id == query_id).first()
    query = preprocess(query_content.translated)        
    content = Entry.query.filter(Entry.id.in_(candidates)).all()
    for _content in content:
        processed = preprocess(_content.content)
        corpus.append(processed)

    indices, similarities = tfidf(query, corpus)
    export = reorder(content, indices, similarities)

    truncate_length = min(5, len(export))
    export = export[:truncate_length]
    return export

def group_by_english(english, translated_others, indices, similarity):
    from collections import defaultdict
    _d = defaultdict(list)
    _id = {}
    for idx, index in enumerate(indices):
        eid = english[idx].id
        oid = translated_others[index].parent_id
        _id[oid] = (eid, similarity[idx])
        t = (oid, similarity[idx])
        _d[eid].append(t)
    return _d, _id


def retrieve_neighbours(query_id):
    english, translated_others = bucket(query_id)
    english_docs = [preprocess(entry.content) for entry in english]
    approx_english_docs = [preprocess(other.translated) for other in translated_others]
    # print(english_docs, approx_english_docs)
    indices, similarities = tfidf(english_docs, approx_english_docs)
    # groups(english, translated_others, indices, similarities)
    entry = Entry.query.get(query_id)
    g, ig = group_by_english(english, translated_others, indices, similarities)

    def construct_retrieved(ls):
        Retrieved = namedtuple('Retrieved', 'id similarity')
        return [
            Retrieved(id=id, similarity=sim) \
            for id, sim in ls
        ]


    if entry.lang == 'en':
        # One routine
        # Return corresponding indices from approx_english_docs
        return construct_retrieved(g[query_id])
    else:
        # Another routine
        en, sim = ig[query_id]
        ls = g[en]
        return construct_retrieved([(en, sim)] + ls)




