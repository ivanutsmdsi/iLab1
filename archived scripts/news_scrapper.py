from gnews import GNews
import pandas as pd
import datetime
import time

gn = GNews()

gn.results = 100
gn.country = 'AU'

s_month = 8
e_month = 9
s_yr = 2022
e_yr = 2022

gn.start_date = (s_yr, s_month, 1)
gn.end_date = (e_yr, e_month, 1)

i = 1
df = pd.DataFrame(columns=['Title','Source','Published','Summary'])

while i <= 100:
    gn.start_date = (s_yr, s_month, 1)
    gn.end_date = (e_yr, e_month, 1)
    
    search = gn.get_news('construction safety')

    print("search done")


    for entry in search:
        title = entry['title']
        source = entry['publisher']
        published = entry['published date']
        summ = entry['description']
        df2 = pd.DataFrame([[title, source, published,summ]],columns=['Title','Source','Published','Summary'])
        df = pd.concat([df, df2], ignore_index=True)
    
    s_month = s_month - 1
    e_month = e_month - 1
    if s_month == 0:
        s_month = 12
        s_yr = s_yr - 1
    if e_month == 0:
        e_month = 12
        e_yr = e_yr - 1
    
    i = i+1

    print("processed step: ", i)
    time.sleep(6)

df.index += 1
outfile = open("news_data.csv","w",newline='',encoding='utf-8')
df.to_csv('news_data.csv')
outfile.close()
print("all saved")