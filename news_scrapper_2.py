import feedparser
import pandas as pd
import sys
from datetime import date
import dateutil.relativedelta

## init a blank pandas df shell to append results to
news_df = pd.DataFrame(columns=['Title','Source','Published'])

##
# save dataframe to output file
##
def save_to_csv():
    save_csv = open("news_output.csv","w",newline='',encoding='utf-8')
    news_df.to_csv('news_output.csv')
    save_csv.close()
    return

##
# returns the pandaframe
##
def get_df():
    return news_df

##
# Save the entries in a feed object to the main news_df dataframe, appending to previous entries
##
def parse_feed_to_df(feed):

    for entry in feed['entries']:
        title = entry['title']
        source = entry['source']['title']
        published = entry['published']
        df2 = pd.DataFrame([[title, source, published]],columns=['Title','Source','Published'])
        news_df = pd.concat([news_df, df2], ignore_index=True)

    return

##
# Collect response from google news based on search patterns
# return the feed object
##

def get_feed(query, start, after):
    params = query + '+before:' + start + '+after:' + after
    addr = 'https://news.google.com/rss/search?q=' + paramscl
    feed = feedparser.feed(addr)
    return feed

##
# Analyse the arguments returned and then build a search pattern against Google News RSS feed
##
def search_news(query, start, after):

    return

def invalid_args():
    print("news_scrapper_2.py error: Not enough arguments found, refer to documentation")
    return

##
#   Based on today's date. get the start date of last month to the start of this month (for date building)
#   Return the dates as strings in a len 2 array
##
def get_prev_month():
    today = date.today()
    end = today.replace(day=1)
    start = end - dateutil.relativedelta.relativedelta(months=1)

    return [str(start),str(end)]

def main():
    #arg definitions:
    #   -lm     search range is last month                  || acts as default if no date range set
    #   -m      search range is month-to-date
    #   -s      read search_query file for query language   
    #   -q      search using query parameter
    #   -csv    save results to csv file                    || acts as default if no save option set (TODO: apply db save as default)
    args = sys.argv[1:]

    if len(args) == 0:
        invalid_args()
    else:
        print("Hello World!")


## Execute main
if __name__ == "__main__":
    main()