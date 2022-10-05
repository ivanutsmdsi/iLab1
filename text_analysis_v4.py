# %% [markdown]
# Text_Analysis_v4
# Process
# - read in and combine
# - remove duplicate 
# - create custom stoplist
# - POS tag
# - text distance
# 
# Outputs
# - output 1 - custom stoplist 
# - output 2 - POS tag list
# - output 3 - Sentitment score
# - output 4 - bi-gram
# - output 5 - Text distance

# %%
# Library
import pandas as pd
import numpy as np
import os        # provides functions necessary to interact with
from pathlib import Path, PureWindowsPath # set folder path
import glob      # used to extract file paths that match specifi
import re        # to remove character not in RE_VALID
import datetime # use timestampe in file name
import unicodedata ## for remove accents function
import string ## for remove accents function

import textblob # for sentiment scoring

import nltk
#nltk.download('stopwords') # generic stop words
#nltk.download('omw-1.4') # use in stemming
#nltk.download('punkt') # sentence tokenizer
from nltk.corpus import stopwords # generic stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS #for text distance
from gensim.models import Word2Vec

# %%
# clean variables
path = []
input_folder_=[]
output_folder_ =[]
format_=[]
all_files = []
raw_file_path=[]
output_file_path=[]
li = []
df0=[]
df=[]

# %%
# Read in raw .csvs from folder to df
# get current work path, where this .py locate
path = os.getcwd()
input_folder_ = 'output_raw'
output_folder_ = 'output_analysis'
format_ = '.csv'

# read in folder path - dynamic
raw_file_path = os.path.join(path, input_folder_)
output_file_path = os.path.join(path, output_folder_)


print(f'Read in folder path: {raw_file_path}')
print('========================================')

print(f'Output folder path: {output_file_path}')
print('========================================')

# get list of all .csvs in the folder
os.chdir(raw_file_path)
all_files = glob.glob(str(raw_file_path) + '/*.csv')
print(f'.csv to read in : {all_files}')
print('========================================')

# %%
# create blank df and read in all .csv in the folder
print(f'Reading in .csv...')
li = []
for filename in all_files:
    df0 = pd.read_csv(filename, index_col=None, header = 0)
    li.append(df0)

df0 = pd.concat(li,axis = 0, ignore_index = True)

#df0.head()
print('========================================')

# %%
print(f'Transform text type and remove duplicate records...')

df0['query_pattern'] = df0['query_pattern'].str.replace(r'q=', '').str.replace(r'\%20',' ')

# convert all possible float to string
df0['title'] = df0['title'].astype(str)
df0['abstract'] = df0['abstract'].astype(str)

print('Convert GMT datetime to Sydney datetime...')
df0.rename(columns={ 'Unnamed: 0':'Origin_id'}, inplace=True)
p_format = "%Y-%m-%dT%H:%M:%SZ"
g_format = "%a, %d %b %Y %H:%M:%S GMT"

df0['published'] = pd.to_datetime(df0['published'], format = p_format, errors = "coerce").fillna(pd.to_datetime(df0['published'], format = g_format, errors = "coerce"))


print('========================================')

print(f'Total {df0.shape[0]} records with {df0.shape[1]} columns from the original input.')
#df0.head()
print('========================================')

# %%
# remove duplicate by title and source columns
df = df0.drop_duplicates(
  subset = ['title', 'source','abstract'],
  keep = 'last').reset_index(drop = True)
df['year'] = df.published.dt.year
df['month'] = df.published.dt.month
df['yyymm'] = df.published.dt.to_period('M')
print(f'Total {df0.shape[0]-df.shape[0]} duplicate records (same title & source & abstract) been removed.')

print('========================================')

# add index record number
print('Creating index numbers in the combined df...')
df['rec_id'] = np.arange(1, df.shape[0] + 1)

print('========================================')

# %%
# create initial stop words
stops = set(stopwords.words('english'))

cust_stop = ['conclusions:', 'introduction:', 'methods:', 'purpose:', 'results:', 'objective:','background:']
stops.update(['br', 'href','\n'],cust_stop)

# %%
# clean columns for title and abstract
## ([^A-Za-z- \t\.]) <- keep alphabetical characters and hyphens, full stop
## ( - )   <- discard hyphen when it is on its own
## to check df[df.custom_abstract.str.contains('._')]
print('Removing numeric and symbols...')

df['custom_title'] = df['title'].str.replace(r'\\n', '').replace(r'\!', '.').replace(r'\._',' ').str.lower()
df['custom_title'] = df['custom_title'].apply(lambda x: re.sub(r"([^A-Za-z- \t\.])|( - )", "", x))

df['custom_abstract'] = df['abstract'].str.replace(r'\\n', '').replace(r'\!', '.').replace(r'\._',' ').str.lower()
df['custom_abstract'] = df['custom_abstract'].apply(lambda x: re.sub(r"([^A-Za-z- \t\.])|( - )", "", x))

print('Removing stop words...')
## clean up extra whitespaces if present
# "\s\s+" <- looks for trailing whitespaces
# trim whitespaces at edges
# remove stopwords
df['custom_title'] = df['custom_title'].apply(lambda x: re.sub(r"\s\s+", " ", x)).str.strip().apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
df['custom_abstract'] = df['custom_abstract'].apply(lambda x: re.sub(r"\s\s+", " ", x)).str.strip().apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
print('========================================')

# %%
# remove stop words by tf-idf
## in title
custom_title= df['custom_title']

custom_abstract= df['custom_abstract']

uni_count = CountVectorizer(stop_words=list(stops) 
                           ,ngram_range=(1,1) 
                           )

uni_word_count=uni_count.fit_transform(custom_title)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(uni_word_count)

## convert the transformer to idf value
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=uni_count.get_feature_names_out(),columns=["idf_weights"])
print(f'IDF table - title:')
print(df_idf.sort_values(by=['idf_weights']))
print('========================================')

print('Distribution of the idf_weights')
print(df_idf.idf_weights.describe())
print('========================================')

## decide how many words to removes
idf_stop = df_idf.idf_weights.describe()['std'] + df_idf.idf_weights.describe()['min'] 
print(f'words with low idf_weights across all docs, total {len(df_idf[df_idf.idf_weights<idf_stop])} words.')
print(df_idf[df_idf.idf_weights<idf_stop].sort_values(by=['idf_weights']))
print('========================================')

idf_stop_list = list(df_idf[df_idf.idf_weights<idf_stop].sort_values(by=['idf_weights']).index)


## whether add to custom stop list
_to_add_stop_words = 'Y'

if _to_add_stop_words == 'Y':
    stops_title = list(stops) + list(idf_stop_list)
else:
    stops_title = stops

print(f'There are {len(stops_title)} final stop words for title.')
print(stops_title)
print('========================================')

# %%
# remove stop words by tf-idf
## In abstract
uni_count_abs = CountVectorizer(stop_words=list(stops) 
                           ,ngram_range=(1,1) 
                           )

uni_word_count_abs=uni_count_abs.fit_transform(custom_abstract)
tfidf_transformer_abs=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer_abs.fit(uni_word_count_abs)

## convert the transformer to idf value
df_idf_abs = pd.DataFrame(tfidf_transformer_abs.idf_, index=uni_count_abs.get_feature_names_out(),columns=["idf_weights"])
print(f'IDF table - abstract:')
print(df_idf_abs.sort_values(by=['idf_weights']))
print('========================================')

print('Distribution of the idf_weights')
print(df_idf_abs.idf_weights.describe())
print('========================================')


## decide how many words to removes
idf_stop_abs = df_idf_abs.idf_weights.describe()['std'] + df_idf_abs.idf_weights.describe()['min'] 
print(f'words with low idf_weights across all docs, total {len(df_idf_abs[df_idf_abs.idf_weights<idf_stop_abs])} words.')
print('========================================')

print(df_idf_abs[df_idf_abs.idf_weights<idf_stop_abs].sort_values(by=['idf_weights']))
idf_stop_list_abs = list(df_idf_abs[df_idf_abs.idf_weights<idf_stop_abs].sort_values(by=['idf_weights']).index)
print('========================================')


## whether add to custom stop list
_to_add_stop_words = 'Y'

if _to_add_stop_words == 'Y':
    stops_abs = list(stops) + list(idf_stop_list_abs)
else:
    stops_abs = stops

print(f'There are {len(stops_abs)} final stop words for abstract.')
print(stops_abs)
print('========================================')

# %%
##  remove custom_stop list from the abstract
print(f'Remove custom stop words in abstract...')

custom_abstract_2 = df['custom_abstract'].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stops_abs))
df['custom_abstract'] = custom_abstract_2 

print('========================================')

##  remove custom_stop list from the title
print(f'Remove custom stop words in title...')

custom_title_2 =custom_title.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stops_title))
df['custom_title'] = custom_title_2

print('========================================')

# %%
## output 1 A - stoplist for title, file save to 
filename1a = '\\output_title_stop_words_'+ datetime.datetime.now().strftime("%Y%m%d")+'.csv'
pd.DataFrame(list(stops_title)).to_csv(str(output_file_path)+ filename1a, index=True)
print (f'Stop words for title are saved as {str(output_file_path)+ filename1a}.')
print('========================================')

## output 1 b - stoplist for _abs, file save to 
filename1b = '\\output_abstract_stop_words_'+ datetime.datetime.now().strftime("%Y%m%d")+'.csv'
pd.DataFrame(list(stops_abs)).to_csv(str(output_file_path)+ filename1b, index=True)
print (f'Stop words for abstract are saved as {str(output_file_path)+ filename1b}.')
print('========================================')

# %%
# POS taging
# tokenize the text. preparing for POS tagging
clean_title= df['custom_title'].tolist()
clean_abstract= df['custom_abstract'].tolist()
stops_title = stops_title
stops_abs = stops_abs

# %%
## functions to tokenize 
def tokenize_sent(df_):
    if df_ is None:
        return None
    elif isinstance(df_, str):
        return sent_tokenize(df_)
    elif isinstance(df_, list):
        return [tokenize_sent(i) for i in df_]
    else:
        return df_ 


def tokenize_wd(df_):
    if df_ is None:
        return None
    elif isinstance(df_, str):
        return word_tokenize(df_)
    elif isinstance(df_, list):
        return [tokenize_wd(i) for i in df_]
    else:
        return df_ 

# %%
print('Applying tokenizer to abstract...')
tokenize_sent_abs = tokenize_sent(clean_abstract)
tokenize_sent_words_list_abs = tokenize_wd(tokenize_sent_abs)
print('========================================')

print('Applying tokenizer to title...')
tokenize_sent_title = tokenize_sent(clean_title)
tokenize_sent_words_list_title = tokenize_wd(tokenize_sent_title)
print('========================================')

# %%
# func: Remove row within df by cell value
def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

# initial stemer and lemmatizer
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# %%
# Tag title
print('Tagging titles, this will take a while... A full list of the POS tags are in https://pythonexamples.org/nltk-pos-tagging/')
li_tokens_title=[]
for i in range(len(tokenize_sent_words_list_title)):
    for j in range(len(tokenize_sent_words_list_title[i])):        
            for l in range(len(tokenize_sent_words_list_title[i][j])):
                stem =stemmer.stem(tokenize_sent_words_list_title[i][j][l])
                lem = lemmatizer.lemmatize(tokenize_sent_words_list_title[i][j][l])  
                org_word = tokenize_sent_words_list_title[i][j][l]
                li_tokens_title.append((j,i,org_word,stem,lem)) 
li_pos_title=[]
for i in range(len(tokenize_sent_words_list_title)):
    for j in range(len(tokenize_sent_words_list_title[i])):
            for tok, tag in nltk.pos_tag(tokenize_sent_words_list_title[i][j]):
                li_pos_title.append((j,i,tok, tag))   

print('=======================================')

print('Converting the tagged list to df...')
df_title_pos_0 = pd.DataFrame (li_tokens_title, columns = ['x','rec_id', 'title_word','stem','lem'])
df_title_pos_1 = pd.DataFrame (li_pos_title, columns = ['x','rec_id', 'title_word','POS'])
df_title_pos = pd.merge(df_title_pos_0, df_title_pos_1,  how='left', left_on=['rec_id','title_word'], right_on = ['rec_id','title_word']).drop_duplicates( subset = ['rec_id', 'title_word','POS'],
 keep = 'last').reset_index(drop = True)

df_title_pos.rec_id = df_title_pos.rec_id+1
print('=======================================')

# drop unused column and text value
df_title_pos = df_title_pos.drop(columns = ['x_x','x_y'])
df_title_pos = filter_rows_by_values(df_title_pos, "title_word", ["[","]","'",".","nan"])

# %%
# Tag abstract 
print('Tagging abstracts, this will take a while... A full list of the POS tags are in https://pythonexamples.org/nltk-pos-tagging/')
li_tokens_abs=[]
for i in range(len(tokenize_sent_words_list_abs)):
    for j in range(len(tokenize_sent_words_list_abs[i])):
        for l in range(len(tokenize_sent_words_list_abs[i][j])):
                stem =stemmer.stem(tokenize_sent_words_list_abs[i][j][l])
                lem = lemmatizer.lemmatize(tokenize_sent_words_list_abs[i][j][l])  
                org_word = tokenize_sent_words_list_abs[i][j][l]
                li_tokens_abs.append((j,i,org_word,stem,lem)) 
li_pos_abs=[]
for i in range(len(tokenize_sent_words_list_abs)):
    for j in range(len(tokenize_sent_words_list_abs[i])):
            for tok, tag in nltk.pos_tag(tokenize_sent_words_list_abs[i][j]):
                li_pos_abs.append((j,i,tok, tag))  


print('Converting the tagged list to df...')
df_abs_pos_0 = pd.DataFrame (li_tokens_abs, columns = ['x','rec_id', 'abstract_word','stem','lem'])
df_abs_pos_1 = pd.DataFrame (li_pos_abs, columns = ['x','rec_id', 'abstract_word','POS'])


df_abs_pos = pd.merge(df_abs_pos_0, df_abs_pos_1,  how='left', left_on=['rec_id','abstract_word'], right_on = ['rec_id','abstract_word']).drop_duplicates( subset = ['rec_id', 'abstract_word','POS'],
 keep = 'last').reset_index(drop = True)

df_abs_pos.rec_id = df_abs_pos.rec_id+1
print('=======================================')

# drop unused column and text value
df_abs_pos = df_abs_pos.drop(columns = ['x_x','x_y'])
df_abs_pos = filter_rows_by_values(df_abs_pos, "abstract_word", ["[","]","'",".","nan"])

# %%
# sace file as .csv
## output 2a - words with POS tag, file save to 
filename2a = '\\output_title_tag_'+ datetime.datetime.now().strftime("%Y%m%d")+'.csv'
pd.DataFrame(df_title_pos).to_csv(str(output_file_path)+ filename2a, index=True)
print (f'Tagged Title are saved as {str(output_file_path)+ filename2a}.')
print('========================================')

filename2b = '\\output_abstract_tag_'+ datetime.datetime.now().strftime("%Y%m%d")+'.csv'
pd.DataFrame(df_abs_pos).to_csv(str(output_file_path)+ filename2b, index=True)
print (f'Tagged Abstract are saved as {str(output_file_path)+ filename2b}.')
print('========================================')

# %%
# Sentiment Scoring
lst_title = list(np.arange(0,int(len(clean_title))))
lst_abs = list(np.arange(0,int(len(clean_abstract))))

# %%
# Add in sentiment score per entry of title
print('Sentiment scoring clean title...')
sentiment_score_title = []
for i in lst_title:
  sentiment_score_title.append(textblob.TextBlob(str(clean_title[i])).sentiment.polarity) 
print('========================================')

# Add in sentiment score per entry of abstract
print('Sentiment scoring clean abstract...')
sentiment_score_abs = []
for i in lst_abs:
  sentiment_score_abs.append(textblob.TextBlob(str(clean_abstract[i])).sentiment.polarity) 
print('========================================')

df['sentiment_score_title'] = sentiment_score_title
df['sentiment_score_abstract'] = sentiment_score_abs

# %%
print('========================================')
## output 3 - file with sentitment score, file save to 
filename3 = '\\output_sentiment_score_'+ datetime.datetime.now().strftime("%Y%m%d")+'.csv'
pd.DataFrame(df).to_csv(str(output_file_path)+ filename3, index=True)
print (f'Extraction with sentiment score are saved as {str(output_file_path)+ filename3}.')
print('========================================')

# %%
# bi-gram extraction
print('========================================')
n = 2

print('Split title to bi-grams...')
df['custom_title'] = df['custom_title'].replace('- ', ' ', regex=True).replace('\._ ', ' ', regex=True).tolist()
df['title_bigrams'] = df['custom_title'].str.replace(r'\[|\]','', regex=True).str.split().apply(lambda x: list(map(' '.join, nltk.ngrams(x, n=n))))

new_df_title = pd.DataFrame(df.title_bigrams.values.tolist(), index=df.rec_id).stack()
new_df_title = new_df_title.reset_index([0, 'rec_id'])
new_df_title.columns = ['rec_id', 'title_bigrams']
new_df_title['title_bigrams']=new_df_title['title_bigrams'].apply(lambda x: re.sub(r"([^A-Za-z- \t])|( - )", "", x))
new_df_title = pd.merge(new_df_title, df[['rec_id','label','yyymm']],  how='left', left_on=['rec_id'], right_on = ['rec_id'])

# %%
new_df_title

# %%
# split abstract as bigrams
print('========================================')
print('Split abstract to bi-grams...')
df['custom_abstract'] = df['custom_abstract'].replace('- ', ' ', regex=True)
df['abstract_bigrams0'] = df['custom_abstract'].str.replace(r'\[|\]','', regex=True).str.split().apply(lambda x: list(map(' '.join, nltk.ngrams(x, n=n))))
df['abstract_bigrams'] = df['abstract_bigrams0'].replace(r'\.|\._ ', '')
new_df_abstract = pd.DataFrame(df.abstract_bigrams.values.tolist(), index=df.rec_id).stack()
new_df_abstract = new_df_abstract.reset_index([0, 'rec_id'])
new_df_abstract.columns = ['rec_id', 'abstract_bigrams']
new_df_abstract['abstract_bigrams']=new_df_abstract['abstract_bigrams'].apply(lambda x: re.sub(r"([^A-Za-z- \t])|( - )", "", x))
new_df_abstract = pd.merge(new_df_abstract, df[['rec_id','label','yyymm']],  how='left', left_on=['rec_id'], right_on = ['rec_id'])

# %%
new_df_abstract

# %%
print('========================================')
## output 4 - Titles split into bigrams with rec_id, file save to 
filename4a = '\\output_title_bigrams_'+ datetime.datetime.now().strftime("%Y%m%d")+'.csv'
pd.DataFrame(new_df_title).to_csv(str(output_file_path)+ filename4a, index=True)
print (f'Title bigrames are saved as {str(output_file_path)+ filename4a}.')
print('========================================')

filename4b = '\\output_abstract_bigrams_'+ datetime.datetime.now().strftime("%Y%m%d")+'.csv'
pd.DataFrame(new_df_abstract).to_csv(str(output_file_path)+ filename4b, index=True)
print (f'Abstract bigrames are saved as {str(output_file_path)+ filename4b}.')
print('========================================')

# %%
# Text distance 
## compare between label/industry

li_all_labels = df['label'].unique().tolist()
#li_all_labels.remove(np.nan)
print(f'Total {len(li_all_labels)} industries:{li_all_labels}.')

# %%
def custom_split_lists(df_col_keep,df_col_label, label_list,sufix_name):
    for i in range(len(label_list)):
     globals()[f'clean_{sufix_name}_{label_list[i]}'] = df_col_keep[df_col_label==label_list[i]].tolist()
     globals()[f'ls_{sufix_name}_{label_list[i]}']  = list(np.arange(0,int(len(df_col_keep[df_col_label==label_list[i]].tolist()))))
     
    print(f'Created {len(label_list)*2} lists from "lable" column: clean_{sufix_name}_{label_list[0]} and clean_{sufix_name}_{label_list[1]} and ls_{sufix_name}_{label_list[0]} and ls_{sufix_name}_{label_list[1]}.')     

# %%
# list to search from - clean df
print(f'Split lists by industry/label')

print(custom_split_lists(df['custom_title'],df['label'],li_all_labels,'custom_title'))

print('========================================')

print(custom_split_lists(df['custom_abstract'],df['label'],li_all_labels,'custom_abstract'))
print('========================================')

# %%
# list to search for - bi-gram
custom_split_lists(new_df_title['title_bigrams'],new_df_title['label'],li_all_labels,'bigram_title')
print('========================================')

custom_split_lists(new_df_abstract['abstract_bigrams'],new_df_abstract['label'],li_all_labels,'bigram_abstract')
print('========================================')

# %%
# create bigram models base on the clean title records construction
tokenized_title_list_construction = [nltk.word_tokenize(sent) for sent in clean_custom_title_construction]
phrase_model_title_li_construction = Phrases(tokenized_title_list_construction , min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)


# %%
# create bigram models base on the clean abstract records constraction
tokenized_abstract_list_construction = [nltk.word_tokenize(sent) for sent in clean_custom_abstract_construction]
phrase_model_abstract_li_construction = Phrases(tokenized_abstract_list_construction , min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)

# %%
# create bigram models base on the clean title records aged_care
tokenized_title_list_aged_care = [nltk.word_tokenize(sent) for sent in clean_custom_title_aged_care]
phrase_model_title_li_aged_care = Phrases(tokenized_title_list_aged_care , min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)


# %%
# create bigram models base on the clean abstract records constraction
tokenized_abstract_list_aged_care = [nltk.word_tokenize(sent) for sent in clean_custom_abstract_aged_care]
phrase_model_abstract_li_aged_care = Phrases(tokenized_abstract_list_aged_care , min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)

# %%
# Use the bi-gram model to create a list of bi-gram work from the cleaned title df
li_title_phrase_construction =[]
for phrase in phrase_model_title_li_construction[tokenized_title_list_construction]:
   li_title_phrase_construction.append(phrase)

# %%
li_abstract_phrase_construction =[]
for phrase in phrase_model_abstract_li_construction[tokenized_abstract_list_construction]:
   li_abstract_phrase_construction.append(phrase)

# %%
li_title_phrase_aged_care =[]
for phrase in phrase_model_title_li_aged_care[tokenized_title_list_aged_care]:
   li_title_phrase_aged_care.append(phrase)

# %%
li_abstract_phrase_aged_care =[]
for phrase in phrase_model_abstract_li_aged_care[tokenized_abstract_list_aged_care]:
   li_abstract_phrase_aged_care.append(phrase)

# %%
# Apply generic word2vec model to the bi-gram tokenized cleaned title df
print('Applying generic word2vec model to the title construction...')
word2vec_model_title_construction = gensim.models.Word2Vec(li_title_phrase_construction,min_count=5)

# %%
print('Applying generic word2vec model to the abstract construction...')
word2vec_model_abstract_construction = gensim.models.Word2Vec(li_abstract_phrase_construction,min_count=5)

# %%
print('Applying generic word2vec model to the title age_care...')
word2vec_model_title_aged_care = gensim.models.Word2Vec(li_title_phrase_aged_care,min_count=5)

# %%
print('Applying generic word2vec model to the abstract age_care...')
word2vec_model_abstract_aged_care = gensim.models.Word2Vec(li_abstract_phrase_aged_care,min_count=5)
print('=======================================')

# %%
# tokenized the bi-gram list to search for generated in the previous section
tokenized_title_bi_list_construction = [nltk.word_tokenize(word) for word in clean_bigram_title_construction]

# %%
tokenized_abstract_bi_list_construction = [nltk.word_tokenize(word) for word in clean_bigram_abstract_construction]

# %%
tokenized_title_bi_list_aged_care = [nltk.word_tokenize(word) for word in clean_bigram_title_aged_care]

# %%
tokenized_abstract_bi_list_aged_care = [nltk.word_tokenize(word) for word in clean_bigram_abstract_aged_care]

# %%
# function to calculate similarity of the from the bi-gram list to cleaned df
def cal_simularity_score(word2vecmodel, bigram_list):
    li_similar_words=[]
    li_most_similar_words_in_df=[]
    for i in range(len(bigram_list)):
        try:
            words = word2vecmodel.wv.most_similar(bigram_list[i])
            for word in words:
                li_similar_words.append(word)
                li_most_similar_words_in_df.append(((bigram_list[i][0]+' '+bigram_list[i][1])))
        except Exception:
            pass
    df_similarity_0 = pd.DataFrame (li_similar_words, columns = ['most_similar_word', 'score'])
    df_similarity_1 = pd.DataFrame (li_most_similar_words_in_df, columns = ['bi_word'])
    df_similarity = df_similarity_1.join(df_similarity_0)
    return df_similarity

# %%
print('Calcualte distance between top bi-gram with the rest of the titles constraction...')
df_title_similarity_construction = cal_simularity_score(word2vec_model_title_construction, tokenized_title_bi_list_construction)
df_title_similarity_construction['industry'] = 'construction'
df_title_similarity_construction['type'] = 'title'
print('=======================================')

# %%
df_title_similarity_construction

# %%
print('Calcualte distance between top bi-gram with the rest of the abstract constraction...')
df_abstract_similarity_construction = cal_simularity_score(word2vec_model_abstract_construction, tokenized_abstract_bi_list_construction)
df_abstract_similarity_construction['industry'] = 'construction'
df_abstract_similarity_construction['type'] = 'abstract'
print('=======================================')

# %%
print('Calcualte distance between top bi-gram with the rest of the titles aged_care...')
df_title_similarity_aged_care = cal_simularity_score(word2vec_model_title_aged_care, tokenized_title_bi_list_aged_care)
df_title_similarity_aged_care['industry'] = 'aged_care'
df_title_similarity_aged_care['type'] = 'title'
print('=======================================')

# %%
print('Calcualte distance between top bi-gram with the rest of the abstract aged_care...')
df_abstract_similarity_aged_care = cal_simularity_score(word2vec_model_abstract_aged_care, tokenized_abstract_bi_list_aged_care)
df_abstract_similarity_aged_care['industry'] = 'aged_care'
df_abstract_similarity_aged_care['type'] = 'abstract'
print('=======================================')

# %%
dt_similarity_title = pd.concat([df_title_similarity_construction, df_title_similarity_aged_care], axis=0, ignore_index=True)
dt_similarity_title['most_similar_word'] = dt_similarity_title['most_similar_word'].apply(lambda x: re.sub(r"\._|\_.", "", x))

dt_similarity_abstract = pd.concat([df_abstract_similarity_construction, df_abstract_similarity_aged_care], axis=0, ignore_index=True)
dt_similarity_abstract['most_similar_word'] = dt_similarity_abstract['most_similar_word'].apply(lambda x: re.sub(r"\._|\_.", "", x))

# %%
dt_similarity_title

# %%
print('========================================')
## output 5 - bigrams similarities, file save to 
filename5a = '\\output_title_bigrams_similarities'+ datetime.datetime.now().strftime("%Y%m%d")+'.csv'
pd.DataFrame(dt_similarity_title).to_csv(str(output_file_path)+ filename5a, index=True)
print (f'Title bigrames are saved as {str(output_file_path)+ filename5a}.')
print('========================================')

filename5b = '\\output_abstract_bigrams_similarities'+ datetime.datetime.now().strftime("%Y%m%d")+'.csv'
pd.DataFrame(dt_similarity_abstract).to_csv(str(output_file_path)+ filename5b, index=True)
print (f'Abstract bigrames are saved as {str(output_file_path)+ filename5b}.')
print('========================================')

# %%
print('========================================')
print('This is THE END of process')
print('Read in summary:')
print(f'RAW csv/s : {all_files}')
print(f'Read in folder path: {raw_file_path}')
print(f'Total {df0.shape[0]} records with {df0.shape[1]} columns from the original input.')
print(f'Total {df0.shape[0]-df.shape[0]} duplicate records (same title & source & abstract) been removed.')
print(f'Output folder path: {output_file_path}')
print('A full list of the POS tags are in https://pythonexamples.org/nltk-pos-tagging/')
print('========================================')

print('List of all output files:')
print (f'1. Stop words for title are saved as {str(output_file_path)+ filename1a}.')
print (f'2. Stop words for abstracts are saved as {str(output_file_path)+ filename1a}.')
print (f'3. Tagged titles are saved as {str(output_file_path)+ filename2a}.')
print (f'4. Tagged abstracts are saved as {str(output_file_path)+ filename2b}.')
print (f'5. Extraction with sentiment scores on both title and abstract are saved as {str(output_file_path)+ filename3}.')
print (f'6. Bigrames - title are saved as {str(output_file_path)+ filename4a}.')
print (f'7. Bigrames - abstract  are saved as {str(output_file_path)+ filename4b}.')
print (f'8. Bigrames - title similarity  are saved as {str(output_file_path)+ filename5a}.')
print (f'9. Bigrames - abstract similarity are saved as {str(output_file_path)+ filename5b}.')
print('========================================')


