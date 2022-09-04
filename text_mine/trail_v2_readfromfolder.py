#Steps:
# 1) read in file
# 2) custom stop list
# 2.1) prefix words from pubmed ['conclusions', 'introduction', 'methods', 'purpose', 'results']
# 2.2) remove words by frequency (not apply yet, still looking for abnormality)
# 3) steming & remove stop words, remove numberic, remove puntuation
# 4) Apply POS
# 5) save as a combined file

#1) get .csv directly from downloaded file
import os
import pandas as pd

os.chdir("/Users/TinaM/Desktop/TMB_File/UTS_SPR_2022/36102/GiHubRepo/iLab1/text_mine")
path = os.getcwd()
pardir_path = os.path.abspath(os.path.join(path, os.pardir))
raw_name = 'news_data_9600'
##raw_name = 'output_9000_aged_care'
input_folder_ = 'tina_test'
output_folder_ = 'text_mine_output'
format_ = '.csv'

raw_file_path = os.path.join(pardir_path, input_folder_)
output_file_path = os.path.join(pardir_path,output_folder_)

df = pd.read_csv(raw_file_path+'\\'+raw_name+format_)
#df.head(10).to_csv(output_file_path+'\\'+ raw_name +'_df_merge_list.csv', index=True)

# 1.1) add index record number
import numpy as np
df['rec_id'] = np.arange(1, df.shape[0] + 1)

# 1.2) rename columns
##raw_name = 'news_data_9600'
df.rename(columns={ 'Title':'title', 'Source':'source','Published': 'published_date', 'Summary':'abstract'}, inplace=True)

##raw_name = 'output_9000_aged_care'
##df.rename(columns={ 'title':'title', 'journal':'source','publication_date': 'published_date', 'abstract':'abstract'}, inplace=True)
## error in custom_abstract =custom_abstract.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stops))
## AttributeError: 'float' object has no attribute 'split', not sure why



# 1.3) convert time zone from GMT to sydney GMT+10
df['published_date']  = pd.to_datetime(df['published_date']).dt.tz_convert('Australia/Sydney')
print (df['published_date'])


print(f'Standardised column names : {df.columns}')

## 2. Create stopword list
#2) Prep file
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4') # use in stemming

from nltk.corpus import stopwords

# 2.1) view the default stopword list:
stops = set(stopwords.words('english'))

pub_med_cust_stop = ['conclusions:', 'introduction:', 'methods:', 'purpose:', 'results:', 'objective:','background:']

stops.update(['br', 'href'],pub_med_cust_stop)

print(f'There are {len(stops)} words in the default stop words list.')
print(stops)

# 2.2) Create another customer stopword list from the word frequency
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# 2.3a) Remove stopwords from the abstract to see find the Tfidf, so the stopwords won't show up in tfidf
custom_abstract = df['abstract'].str.replace(r'[0-9]+', '')
custom_abstract =custom_abstract.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stops))

print('Abstract after remvoing numeric and default stopwords:')
print(custom_abstract)

# 2.3b) remove numberic from the title to see find the Tfidf
custom_title= df['title'].str.replace(r'[0-9]+', '')

print('Title after remvoing numeric and default stopwords:')
print(custom_title)

# 2.4 A) initiate CountVectorizer for abstract
uni_count = CountVectorizer(stop_words=list(stops) + pub_med_cust_stop
                           ,ngram_range=(1,1) 
                           )

uni_word_count=uni_count.fit_transform(custom_abstract)
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(uni_word_count)
## keep words in pub_med_cust_stop list if it's not follow by ':', as they might be part of the real text

### idf value
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=uni_count.get_feature_names(),columns=["idf_weights"])

print(f'IDF table - Abstract:')
print(df_idf.sort_values(by=['idf_weights']))

print('Distribution of the idf_weights')
print(df_idf.idf_weights.describe())

## 2.5A) set up additional stopwords for abstract from idf_weight 
idf_stop = df_idf.idf_weights.describe()[5] ## idf_weights at 25%

print(f'words with idf_weights less than 25% across all docs, total {len(df_idf[df_idf.idf_weights<idf_stop])} words.')
print(df_idf[df_idf.idf_weights<idf_stop].sort_values(by=['idf_weights']))

idf_stop_list = list(df_idf[df_idf.idf_weights<idf_stop].sort_values(by=['idf_weights']).index)

## 2.5A) whether add to custom stop list
_to_add_stop_words = 'N'

if _to_add_stop_words == 'Y':
    idf_stop_list= idf_stop_list
else:
    idf_stop_list = ''

stops_abs = list(stops) + list(idf_stop_list)

print(f'There are {len(stops_abs)} final stop words for abstract. Including {len(stops)} words from default stopwords and {len(idf_stop_list)} words for low ti_idf weight.')
print(stops_abs)
## not to add low idf weigth words as want to keep 'construction', 'safety'

## output 1 A - stoplist for abstract
pd.DataFrame(list(stops_abs)).to_csv('final_stop_words_abstract.csv', index=True)
print(f'The final stop words used in Abstract is saved as .csv')

# 2.4 B) initiate CountVectorizer for title
# title_uni_count = CountVectorizer(
                        #    ngram_range=(1,1) 
                        #    )
# 
# title_uni_word_count=uni_count.fit_transform(custom_title)
# title_tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# title_tfidf_transformer.fit(title_uni_word_count)
# keep words in pub_med_cust_stop list if it's not follow by ':', as they might be part of the real text
# 
##  2.4 B)idf value
# title_df_idf = pd.DataFrame(title_tfidf_transformer.idf_, index=title_uni_count.get_feature_names(),columns=["idf_weights"])
# 
# print(f'IDF table - Title:')
# print(title_df_idf.sort_values(by=['idf_weights']))
# 
# print('Distribution of the idf_weights')
# print(title_df_idf.idf_weights.describe())
# 
# 2.5 B) set up additional stopwords for abstract from idf_weight 
# title_idf_stop = title_df_idf.idf_weights.describe()[5] ## idf_weights at 25%
# 
# print(f'words with idf_weights less than 25% across all docs, total {len(title_df_idf[title_df_idf.idf_weights<idf_stop])} words.')
# print(title_df_idf[title_df_idf.idf_weights<idf_stop].sort_values(by=['idf_weights']))
# 
# title_idf_stop_list = list(title_df_idf[title_df_idf.idf_weights<idf_stop].sort_values(by=['idf_weights']).index)
# 
#  2.5 B) whether add to custom stop list
# _title_to_add_stop_words = 'N'
# 
# if _title_to_add_stop_words == 'Y':
    # title_idf_stop_list= title_idf_stop_list
# else:
    # title_idf_stop_list = ''
# 
# stops_title = list(title_idf_stop_list)
# 
# print(f'There are {len(stops_title)} final stop words for title. Including  {len(stops_title)} words for low ti_idf weight.')
# print(stops_title)
# not to add low idf weigth words as want to keep 'construction', 'safety'
# 
# output 1 B - stoplist for abstract
# pd.DataFrame(list(stops_title)).to_csv('final_stop_words_title.csv', index=True)
# print(f'The final stop words used in title is saved as .csv')
# 

# 3a) Steming - abstract
stops = stops_abs
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# POS (Parts Of Speech) for: NN = nouns, JJ = adjectives, VB= verbs and RB= adverbs, JJR = adjective, comparative (larger), WRB=wh- adverb (how), WP = wh- pronoun (who), WDT = wh-determiner (that, what)
DI_POS_TYPES = {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r', 'JJR':'c', 'WRB':'b','WP':'p','WDT':'t'} 
POS_TYPES = list(DI_POS_TYPES.keys())

# Constraints on tokens
MIN_STR_LEN = 3
RE_VALID = '[a-zA-Z]' ## only keep alphabet

# convcert the text column to analysis to a list
abstracts= df.abstract.tolist()

print(f'Total of {len(abstracts)} entries.')

print('The first entry of abstracts')
print(abstracts[0])


# 3.1a) set up functions to remove accents
import unicodedata ## for remove accents function
import string ## for remove accents function
import re ## to remove character not in RE_VALID

## function from https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/nlp/nltk_preprocess.ipynb#scro
# Remove accents function
def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters or x == " ")

# Process all quotes
li_tokens = []
li_token_lists = []
li_lem_strings = []

for i,text in enumerate(abstracts):
    # Tokenize by sentence, then by lowercase word
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    # Process all tokens per quote
    li_tokens_quote = []
    li_tokens_quote_lem = []
    for token in tokens:
        # Remove accents
        t = remove_accents(token)

        # Remove punctuation
        t = str(t).translate(string.punctuation)
        li_tokens_quote.append(t)
        
        # Add token that represents "no lemmatization match"
        li_tokens_quote_lem.append("-") # this token will be removed if a lemmatization match is found below

        # Process each token
        if t not in stops:
            if re.search(RE_VALID, t):
                if len(t) >= MIN_STR_LEN:
                    # Note that the POS (Part Of Speech) is necessary as input to the lemmatizer 
                    # (otherwise it assumes the word is a noun)
                    pos = nltk.pos_tag([t])[0][1][:2]
                    pos2 = 'n'  # set default to noun
                    if pos in DI_POS_TYPES:
                      pos2 = DI_POS_TYPES[pos]
                    
                    stem = stemmer.stem(t)
                    lem = lemmatizer.lemmatize(t, pos=pos2)  # lemmatize with the correct POS
                    
                    if pos in POS_TYPES:
                        li_tokens.append((t, stem, lem, pos))

                        # Remove the "-" token and append the lemmatization match
                        li_tokens_quote_lem = li_tokens_quote_lem[:-1] 
                        li_tokens_quote_lem.append(lem)

    # Build list of token lists from lemmatized tokens
    li_token_lists.append(li_tokens_quote)
    
    # Build list of strings from lemmatized tokens
    str_li_tokens_quote_lem = ' '.join(li_tokens_quote_lem)
    li_lem_strings.append(str_li_tokens_quote_lem)
    
# Build resulting dataframes from lists
df_token_lists = pd.DataFrame(li_token_lists) # (100, 549)

print("Tokenized words in the first 5 entries:")
print(df_token_lists.head(5).to_string())

# Replace None with empty string
for c in df_token_lists:
    if str(df_token_lists[c].dtype) in ('object', 'string_', 'unicode_'):
        df_token_lists[c].fillna(value='', inplace=True)

df_lem_strings = pd.DataFrame(li_lem_strings, columns=['lem quote'])

df_lem_strings['rec_id'] = np.arange(1, df_lem_strings.shape[0] + 1)

print()
print("")
print("Tokenized, stemed and lemmatized first 5 entries:")
print(df_lem_strings.head(5).to_string())

## tokeniezed words df = df_token_lists
## tokeniezed entry df = df_lem_strings


#tokeniezed words df = df_token_lists
import numpy as np
df_token_lists['rec_id'] = np.arange(1, df_token_lists.shape[0] + 1)

print(df_token_lists.head())

print (df_token_lists.shape) 

df_token_lists_unpivoted = df_token_lists.melt(id_vars=['rec_id'], var_name='word_order', value_name='tokenized_word').drop(['word_order'], axis=1)

df_token_lists_unpivoted = df_token_lists_unpivoted[df_token_lists_unpivoted['tokenized_word'].str.strip().astype(bool)]

print(df_token_lists_unpivoted.head())
df_token_lists_unpivoted.shape 

## Reference list to token, stem version, lem version and the POS tag,
df_all_words = pd.DataFrame(li_tokens, columns=['token', 'stem', 'lem', 'pos'])
print (f'df2 shape: {df_all_words.shape}') #(49023, 4)

## Create reference list with raw entry record number 'rec_id','NN' is also the default value when POS not found

### drop duplucate tokens
df_all_words2 = df_all_words.drop_duplicates()
print (f'Removed duplicates shape: {df_all_words2.shape}') 
df_all_words2.head()

## join with df_token_lists by token to get the rec_id which is the record id in the raw data
df_all_token_words = df_token_lists_unpivoted.merge(df_all_words2, how = 'left', left_on='tokenized_word',right_on ='token').drop(['token'], axis=1)

print (f'All token words list shape: {df_all_token_words.shape}')
df_all_token_words.head() 

## count words by each entries to see if can summaries the entry a bit more 
df_all_token_words_entry = df_all_token_words.groupby(['rec_id','lem','pos']).size().sort_values(ascending=False).reset_index(name='count')

## words within first entry, not able to summaries by the highest number of each POS, as there are many 'NN' with the same count
print('first entry')
df_all_token_words_entry_1 = df_all_token_words_entry[df_all_token_words_entry['rec_id']==1].sort_values('count',ascending=False)
df_all_token_words_entry_1

print(f'Original data size: {df.shape}')
print(f'lem_string data size: {df_lem_strings.shape}')
print(f'tokenised word data size: {df_all_token_words.shape}')


# output 2A - raw data with tokenize words for PBI word counts visual, and lemeatized entry for PBI word cloud, all join with rec_id
from functools import reduce

dfs = [df,df_lem_strings,df_all_token_words]
#dfs = [df,df_all_token_words]

df_merged2 = reduce(lambda  left,right: pd.merge(left,right,on=['rec_id'],
                                            how='outer'), dfs).fillna('')

df_merged2.to_csv(output_file_path+'\\'+ raw_name +'_df_merge_list.csv', index=True)
print(f'The combinded file data size: {df_merged2.shape}. .csv is saved in folder ') ##(14439, 9)
df_merged2.head()