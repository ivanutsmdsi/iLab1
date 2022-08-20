## dependency:
# https://github.com/scholarly-python-package/scholarly
# pip3 install scholarly
# pip3 install pandas


import pandas as pd
from scholarly import scholarly

search_query = scholarly.search_pubs(query = 'construction health safety')

#data = next(search_query)
#df = pd.json_normalize(data)
#
#print('df.T running')
#df.T
#
#df.to_csv('scholarly_data.csv')

data = next(search_query)
df = pd.json_normalize(data)
i = 1
for data in search_query:
    print(i)
    if i == 20:
            break
    temp = pd.json_normalize(data)
    df = pd.concat([df, temp])
    i = i + 1

df.to_csv('scholarly_data.csv')