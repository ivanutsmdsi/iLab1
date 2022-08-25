import os, json
import pandas as pd
from serpapi import GoogleSearch
from urllib.parse import urlsplit, parse_qsl

params = {
    "api_key": "API_KEY",                            # https://serpapi.com/dashboard retrieve API_KEY
    "engine": "google_scholar",                      
    "q": "construction safety health",               # search query
    "hl": "en",                                      
    "as_ylo": "2012",                                # results by earliest year
    "start": "0"                                     # first page
}

search = GoogleSearch(params)         

#print(type(search))
results = search.get_dict()

#print(type(results))
organic_results = results["organic_results"]

#print(type(organic_results))
print(organic_results)

## extract organic_results list, iterate into df and print to csv
data = organic_results[0]
df = pd.json_normalize(data)
i = 1
for data in organic_results[1:]:
    print(i)
    temp = pd.json_normalize(data)
    df = pd.concat([df, temp])
    i = i + 1

df.to_csv('serpapi_data.csv')