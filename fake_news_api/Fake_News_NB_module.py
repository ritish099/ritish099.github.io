import numpy as np
import pandas as pd
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
stemmer = WordNetLemmatizer()
model, vector, tf_vector = pickle.load(open('Fake_News_NB_Model', 'rb'))

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
NUM_RE = re.compile(' \d+ ')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
   
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    text = re.sub(NUM_RE, ' ', text)
    text = [e for e in text.split(' ') if e not in STOPWORDS and e!='']
    return text


def predictNB(text):
    
    if not text: return None
    
    sample = np.array(text_prepare(text))
    sample = sample.reshape(-1, sample.shape[0])
    sample[0] = [stemmer.lemmatize(e) for e in sample[0]]
    sample = [' '.join(e) for e in sample]

    counts = vector.transform(sample)
    sample = tf_vector.fit_transform(counts)
    predd = model.predict_proba(sample)[0]

    return dict({'Real':predd[0], 'Fake':predd[1]})
