## dependency:
# https://github.com/scholarly-python-package/scholarly
# pip3 install scholarly
# pip3 install pandas


import pandas as pd
from scholarly import scholarly

#execute web scrap
search_query = scholarly.search_pubs(query = 'construction health safety')

# store first result into dataframe to build the dataframe
data = next(search_query)
df = pd.json_normalize(data)

# iterate through the rest of the web scrap results
i = 1  
for data in search_query:
    print(i)
    if i == 20:
            break                   # putting a hard limit so I don't exceed Google HTTP requests
    temp = pd.json_normalize(data)
    df = pd.concat([df, temp])      # append each new result into the dataframe for export later
    i = i + 1

df.to_csv('scholarly_data.csv')