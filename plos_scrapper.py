import requests
import argparse
import pandas as pd
import sys
import math
import time
from datetime import datetime
from datetime import date
import dateutil.relativedelta

## init a blank pandas df shell to append results to
abs_df = pd.DataFrame(columns=['journal','published','title', 'abstract'])

##
# save dataframe to output file
##
def save_to_csv():
    global abs_df
    save_csv = open("plos_output.csv","w",newline='',encoding='utf-8')
    abs_df.to_csv('plos_output.csv')
    save_csv.close()
    print("PLOS search results saved to 'plos_output.csv'")
    return

##
# returns the pandaframe
##
def get_df():
    global abs_df
    return abs_df

##
# Save the entries in a feed object to the main abs_df dataframe, appending to previous entries
##
def parse_response_to_df(docs):
    global abs_df
    
    
    for article in docs:
        title = article['title']
        journal = article['journal']
        published = article['publication_date']
        abstract = article['abstract']
        df2 = pd.DataFrame([[journal, published, title, abstract]],columns=['journal','published','title', 'abstract'])
        abs_df = pd.concat([abs_df, df2], ignore_index=True)

    return

##
# Collect response from PLOS search based on search params
# feed the response through the parser to collect data and save to DF
# iterate through the response until no more response is available
##

def get_plos_results(query, before, after,start_row = 1):
    # (publication_date:[2022-08-01T00:00:00Z TO 2022-08-31T23:59:59Z]) AND (construction+health+safety)
    pub_param = '(publication_date:[' + after + 'T00:00:00Z' + ' TO ' + before + 'T00:00:00Z' + '])'
    join_param = ' AND '
    query_param = '(' + query + ')' + join_param + pub_param

    fields_param = '&fl=id,title,abstract,publication_date,journal'

    start_param = '&start=' + str(start_row)

    params = 'q=' + query_param + fields_param + start_param
    
    # print(params)

    url = 'https://api.plos.org/search?rows=100&' + params
    response = requests.get(url)

    if (response.status_code == 200):
        return response.json()
    else:
        print('PLOS search request errored out: HTTP code [' + response.status_code + ']')
        return

##
# Analyse the arguments returned and then build a search request to PLOS API
# Feed in the search request to extract_
##
def extract_abstracts(q, lm = False, m = False, d_range = False, before = None, after = None):
    global abs_df
    print('[Begin PLOS article extraction]')

    ## if no search range provided, default to last month
    if (m is False and d_range is False):
        lm = True

    ## Set date range and replace before and after                     ## at least one of these must be true (lm is default option)
    if(lm):
        range = get_prev_month()
        before = range['end']
        after = range['start']
    elif(m):
        range = get_curr_month()
        before = range['end']
        after = range['start']
    
    ## Type A: search and iterate through results
    response = get_plos_results(query = q, before = before, after = after)

    if (response is None):
        ## end extractions
        return
    
    numFound = response['response']['numFound']
    currPage = 0
    numPages = math.ceil(numFound/100)

    print('PLOS article search found ' + str(numFound) + ' articles.')

    results = len(response['response']['docs'])
    while (currPage < numPages):
        print('Extracting article search results from PLOS (' + str(currPage + 1) + ' of '+ str(numPages) + ')..      ', end = '')

        ## analyse response first
        parse_response_to_df(response['response']['docs'])

        
        ## iterate to the next page
        ## stop if the search reached the end of results
        currPage = currPage + 1
        if(currPage * 100 >= numFound):
            break

        time.sleep(3)
        response = get_plos_results(query = q, before = before, after = after, start_row = currPage * 100)

         ## end extractions if nothing was returned
        if (response is None):
            return

          
    return

##
#   Print error statments for invalid argument combinations
##
def invalid_args(error):
    match error:
        case 1:
            print("Not enough arguments found, call 'plos_scrapper_2.py --help' for guidance.")
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
    
    ##
    #   Part 1:
    #   Validate arguments and return error messages as required
    ## 
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

    ##
    #   Part 2:
    #   after checking arguments are valid, pass arguments to extract_abstracts to build feed query
    ## 
    extract_abstracts(lm = pa.lm, m = pa.m, d_range = d_range, before = pa.before, after = pa.after, q = pa.query_param)

    ##
    #   Part 3:
    #   Save output to either DB or CSV
    ## 

    ## Version 1: save to csv
    save_to_csv()

## Execute main
if __name__ == "__main__":
    main()