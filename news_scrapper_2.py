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
    global news_df
    save_csv = open("gnews_output.csv","w",newline='',encoding='utf-8')
    news_df.to_csv('gnews_output.csv')
    save_csv.close()
    return

##
# returns the pandaframe
##
def get_df():
    global news_df
    return news_df

##
# Save the entries in a feed object to the main news_df dataframe, appending to previous entries
##
def parse_feed_to_df(feed):
    global news_df
    
    # print('num of entries retrieved: ' + str(len(feed['entries']))) 
    
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

def get_feed(query, before, after):
    params = query + '+before:' + before + '+after:' + after
    addr = 'https://news.google.com/rss/search?q=' + params
    feed = feedparser.parse(addr)
    return feed

##
# Analyse the arguments returned and then build a search pattern against Google News RSS feed
##
def get_news(q, lm = False, m = False, d_range = False, before = None, after = None):
    global news_df

    ## if no search range provided, default to last month
    if (m is False and d_range is False):
        lm = True

    ## Set date range                               ## at least one of these must be true (lm is default option)
    if(lm):
        range = get_prev_month()
    elif(m):
        range = get_curr_month()
    ## TODO: search by custom date range -- requires recurising through before and after
    elif(d_range):
        range = {'start': after, 'end': before}


    ## Version 1: assume that every search request is 1 month long
    print('Requesting search results from google news (1 of 1)...')
    feed = get_feed(query = q, before = range['end'], after = range['start'])   ## Execute one get request
    parse_feed_to_df(feed)                                                      ## parse feed into news_df

    return

##
#   Print error statments for invalid argument combinations
##
def invalid_args(error):
    match error:
        case 1:
            print("Not enough arguments found, call 'news_scrapper_2.py --help' for guidance.")
        case 2:
            print("cannot use -lm (last month) or -m (current month) or -before & -after together. Refer to documentation for guidance.")
        case 3:
            print("--before and --after must be used together.")
    return

##
#   Based on the input date. get the start date of last month to the start of this month (for date building)
#   if no date provided, use today's date
#   Return the dates as strings in a len 2 array
##
def get_prev_month(date = date.today()):
    end = date.replace(day=1)
    start = end - dateutil.relativedelta.relativedelta(months=1)

    return {'start': str(start), 'end': str(end)}

##
#   Based on the input date. get the start date of the current month to the start of next month (for date building)
#   if no date provided, use today's date
#   Return the dates as strings in a len 2 array
##
def get_curr_month(date = date.today()):
    start = date.replace(day=1)
    end = start + dateutil.relativedelta.relativedelta(months=1)

    return {'start': str(start), 'end': str(end)}

def main():
    #arg definitions:
    #   -lm     search range is last month                  || acts as default if no date range set
    #   -m      search range is month-to-date
    #   -o      read search_query file for query language   
    #   -q      search using query parameter
    #   -csv    save results to csv file                    || acts as default if no save option set (TODO: apply db save as default)
    args = sys.argv[1:]                                     ## replace sys.argv with argparse
    d_range = False                                         ## flag for search by date range

    print('[Begin Google News article title extraction]')

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
    if (pa.query_csv_file is not None):
        print("Search query input by csv file not implemented yet. Please use -q")
        return
    elif (pa.query_param is None):
        print("Search query required.")
        return

    ## after checking arguments are valid, pass arguments to search_news to build feed query
    get_news(lm = pa.lm, m = pa.m, d_range = d_range, before = pa.before, after = pa.after, q = pa.query_param)


    ## Save results

    ## Version 1: save to csv
    save_to_csv()

## Execute main
if __name__ == "__main__":
    main()