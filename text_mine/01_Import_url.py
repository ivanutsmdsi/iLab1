#1) get .csv directly from github URL
import requests
import io
url = "https://raw.githubusercontent.com/ivanutsmdsi/iLab1/william/Output/pubmed_data100.csv"
s=requests.get(url).content

## import pubmed_data.csv---
import pandas as pd
df=pd.read_csv(io.StringIO(s.decode('utf-8')))

## subset the data with abstract column only
text = df['Abstract']

print(text)

#2) Prep file
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 
nltk.download('omw-1.4') # use in stemming

from nltk.corpus import stopwords

# 2.1) Create stopword list:
stops = set(stopwords.words('english'))
stops.update(['br', 'href','Introduction:','Methods: ','Results:','Conclusions:','Purpose:'])
print(f'There are {len(stops)} words in the default stop words list.')
print(stops)

# 2.2) Create another customer stopword list from the word frequency
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

## 2.2.1) using the count vectorizer to generate a list of word vs weights
count = CountVectorizer()
word_count=count.fit_transform(df['Abstract'])

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count)
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=count.get_feature_names_out(),columns=["idf_weights"])

df_idf.sort_values(by=['idf_weights'])

tf_idf_vector=tfidf_transformer.transform(word_count)
feature_names = count.get_feature_names_out()

first_document_vector=tf_idf_vector[1]
df_tfifd= pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])

df_tfifd.sort_values(by=["tfidf"],ascending=True)

## 2.2.2) mannually inspect the the list
df_tfifd.sort_values(by=["tfidf"],ascending=True).to_csv('vectorized_word_list.csv', index=True)