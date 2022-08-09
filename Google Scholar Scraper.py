# Attempt to web scrape Google Scholar
# Link based on Medium Article "Scraping Google Scholar with Python and BeautifulSoup"
# Source: https://proxiesapi-com.medium.com/scraping-google-scholar-with-python-and-beautifulsoup-850cbdfedbcf

from bs4 import BeautifulSoup
import requests

headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36'}
url = 'https://scholar.google.com.au/scholar?hl=en&as_sdt=0%2C5&q=work+safety&btnG='

response = requests.get(url, headers=headers)
soup= BeautifulSoup(response.content, 'lxml')

# print(soup.select('[data-lid]'))
#for item in soup.select('[data-lid]'):
#    try:
#        print('----------------------------------------')
#        print(item)

#    except Exception as e:
#        # raise e
#        print('')

#print(soup.select('[data-lid]'))
#for item in soup.select('[data-lid]'):
#    try:
#        print('----------------------------------------')
#        # print(item)
#        print(item.select('h3')[0].get_text())
#    except Exception as e:
#        # raise e
#        print('')

print(soup.select('[data-lid]'))
for item in soup.select('[data-lid]'):
    try:
        print('----------------------------------------')
        #print(item)
        print(item.select('h3')[0].get_text())
        print(item.select('a')[0]['href'])
        print(item.select('.gs_rs')[0].get_text())
        print('----------------------------------------')
    except Exception as e:
        #raise e
        print('')

