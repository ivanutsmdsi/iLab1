#1) get .csv directly from github URL
import requests
import io
url = "https://raw.githubusercontent.com/ivanutsmdsi/iLab1/william/Output/pubmed_data100.csv"
s=requests.get(url).content

## import pubmed_data.csv---
import pandas as pd
import numpy as np

df=pd.read_csv(io.StringIO(s.decode('utf-8')))
#df.rename(columns={'Unnamed: 0':"rec_id" }, inplace=True)
df['rec_id'] = np.arange(1, df.shape[0] + 1)

## before change
print('sample of supplied file')
print(df.head(5))
print(f'size of the supplied file {df.shape}')


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

## output 1 - stop list
## write the stopwords into a .csv

pd.DataFrame(list(stops)).to_csv('final_stop_words.csv', index=True)

# 4) Steming
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# POS (Parts Of Speech) for: NN = nouns, JJ = adjectives, VB= verbs and RB= adverbs, JJR = adjective, comparative (larger), WRB=wh- adverb (how), WP = wh- pronoun (who), WDT = wh-determiner (that, what)
DI_POS_TYPES = {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r', 'JJR':'c', 'WRB':'b','WP':'p','WDT':'t'} 
POS_TYPES = list(DI_POS_TYPES.keys())

# Constraints on tokens
MIN_STR_LEN = 3
RE_VALID = '[a-zA-Z]' ## only keep alphabet


# 4.1) convcert the text column to analysis to a list, the column name has to be 'Abstract'
abstracts= df.Abstract.tolist()

print(f'Total of {len(abstracts)} entries.')

print('The first entry of abstracts')
print(abstracts[0])

# 4.2) set up functions to remove accents
import unicodedata ## for remove accents function
import string ## for remove accents function
import re ## to remove character not in RE_VALID

## function from https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/nlp/nltk_preprocess.ipynb#scrollTo=-44aMwUcQZxm
# Remove accents function
def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters or x == " ")

print("Start to tokenized, stemed and lemmatized on the raw text...")

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
        li_tokens_quote_lem.append("-") # this token will be removed if a lemmatization match is foun

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

print()
print("")
print("Tokenized, stemed and lemmatized words, and complie back to string.")


#tokeniezed words df = df_token_lists

df_token_lists['rec_id'] = np.arange(1, df_token_lists.shape[0] + 1)

print(df_token_lists.head())

print (df_token_lists.shape) # (100, 550)

df_token_lists_unpivoted = df_token_lists.melt(id_vars=['rec_id'], var_name='word_order', value_name='tokenized_word').drop(['word_order'], axis=1)

df_token_lists_unpivoted = df_token_lists_unpivoted[df_token_lists_unpivoted['tokenized_word'].str.strip().astype(bool)]

print (f'total {df_token_lists_unpivoted.shape[0]} words have been tokenised.')

# 4.2) Append POS tag to the tokenised list 
df_all_words = pd.DataFrame(li_tokens, columns=['token', 'stem', 'lem', 'pos'])

df_all_words2 = df_all_words.drop_duplicates()

## join with df_token_lists by token to get the rec_id which is the record id in the raw data
df_all_token_words = df_token_lists_unpivoted.merge(df_all_words2, how = 'left', left_on='tokenized_word',right_on ='token').drop(['token'], axis=1)

#tokeniezed entry df = df_lem_strings
df_lem_strings['rec_id'] = np.arange(1, df_lem_strings.shape[0] + 1)
df_lem_strings.rename(columns={ df_lem_strings.columns[0]: 'lemmatized entry'}, inplace=True)



## output 2 - raw data with tokenize words for PBI word counts visual, and lemeatized entry for PBI word cloud, all join with rec_id
from functools import reduce

dfs = [df,df_lem_strings,df_all_token_words]

df_merged2 = reduce(lambda  left,right: pd.merge(left,right,on=['rec_id'],
                                            how='outer'), dfs).fillna('')

df_merged2.to_csv('df_merge_list.csv', index=True)
print(f'The combined tokenlise word with raw data .csv is saved. The df is {df_merged2.shape[0]} rows with {df_merged2.shape[1]} columns.') ##(14439, 9)
