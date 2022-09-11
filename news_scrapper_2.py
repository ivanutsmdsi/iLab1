import argparse
import feedparser
import pandas as pd
import sys
from datetime import datetime
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
    addr = 'https://news.google.com/rss/search?q=' + params
    feed = feedparser.feed(addr)
    return feed

##
# Analyse the arguments returned and then build a search pattern against Google News RSS feed
##
def search_news(query, start, after):

    return

def invalid_args(error):
    match error:
        case 1:
            print("Not enough arguments found, refer to documentation for guidance.")
        case 2:
            print("cannot use -lm (last month) or -m (current month) or -before & -after together. Refer to documentation for guidance.")
        case 3:
            print("--before and --after must be used together.")
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
    #   -o      read search_query file for query language   
    #   -q      search using query parameter
    #   -csv    save results to csv file                    || acts as default if no save option set (TODO: apply db save as default)
    args = sys.argv[1:]                                     ## replace sys.argv with argparse
    d_range = False                                         ## flag for search by date range

    ## collect arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-lm', action='store_true', help = 'set search range to last month (default value)')
    parser.add_argument('-m', action='store_true' , help = 'set search range to current month to date')
    parser.add_argument('-o', dest='query_csv_file', type =str, help = 'load query terms from csv file named')
    parser.add_argument('-q', dest='query_param',  type =str, help = 'search by query terms provided')
    parser.add_argument('-csv', action='store_true', help = 'save output to csv file')
    parser.add_argument('--before', dest='before', type=lambda d: datetime.strptime(d, '%Y-%m-%d'), help = 'date input must use the format YYYY-MM-DD')
    parser.add_argument('--after', dest='after', type=lambda d: datetime.strptime(d, '%Y-%m-%d'), help = 'date input must use the format YYYY-MM-DD')
    
    pa = parser.parse_args()
    
    
    if len(args) == 0:
        invalid_args(1)
        return

    ## check number of date options used are valid
    d = 0
    if (pa.lm):
        d = d + 1
    if (pa.m):
        d = d + 1
    if (pa.after != None or pa.before != None):
        d = d + 1
    
    if d > 1:
        invalid_args(2)
        return

    ## check both after and before were added
    if (pa.before != None and pa.after == None) or (pa.before == None and pa.after != None):
        invalid_args(3)
        return
    elif pa.before != None and pa.after != None:
        d_range = True      ## both before and after dates were provided


    ## TODO: read query from either csv or argument
    # Check if a search query has been provided
    if (pa.o is not None):
        print("Search query input by csv file not implemented yet. Please use -q")
        return
    elif (pa.q is None):
        print("Search query required.")
        return

    ## after checking arguments are valid, pass arguments to search_news to build feed query
    search_news(lm = pa.lm, m = pa.m, d_range = d_range, before = pa.before, after = pa.after, q = pa.q)


## Execute main
if __name__ == "__main__":
    main()