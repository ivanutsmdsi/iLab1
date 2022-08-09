# Attempt to web scrape Google News

from bs4 import BeautifulSoup
import requests

headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36'}
url = 'https://www.google.com/search?q=workplace+safety&biw=1278&bih=969&tbm=nws&ei=twPxYpyxIIHi4-EPs6ia2A4&ved=0ahUKEwjc6Ny3obf5AhUB8TgGHTOUBusQ4dUDCA0&uact=5&oq=workplace+safety&gs_lcp=Cgxnd3Mtd2l6LW5ld3MQAzILCAAQgAQQsQMQgwEyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQ6BAgAEEM6BggAEB4QFjoICAAQgAQQsQM6CAgAELEDEIMBOgUIABCxAzoLCAAQsQMQgwEQkQI6BQgAEJECUK0CWPcLYM8NaABwAHgAgAGwAYgBtQqSAQMwLjmYAQCgAQHAAQE&sclient=gws-wiz-news'

response = requests.get(url, headers=headers)
soup= BeautifulSoup(response.content, 'lxml')

print(soup.select('[data-ved]'))
for item in soup.select('[data-ved]'):
    try:
        print('----------------------------------------')
        #print(item)
        print(item.select('CEMjEf NUnG9d')[0].get_text())
        #print(item.select('a')[0]['href'])
        #print(item.select('.gs_rs')[0].get_text())
        #print('----------------------------------------')
    except Exception as e:
        #raise e
        print('')