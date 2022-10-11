import argparse
import sys
import time
import runpy
from os.path import exists
from datetime import datetime
from plos_scrapper import extract_abstracts_by_filequery as plos_scrapper_query_csv
from plos_scrapper import extract_abstracts_and_save as plos_scrapper_query
from news_scrapper_2 import get_news_by_filequery as gnews_scrapper_query_csv
from news_scrapper_2 import get_news_and_save as gnews_scrapper_query



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

    
    ## After checks are cleared run all local scrapper py scripts uing the parameters supplied

    ## Check if it is csv query or file query
    if (pa.query_csv_file is not None):
        if (exists(pa.query_csv_file) is False):
            sys.exit("'" + pa.query_csv_file + "' could not be found.")
        
        gnews_scrapper_query_csv(filename = pa.query_csv_file, 
                                        lm = pa.lm, 
                                        m = pa.m, 
                                        d_range = d_range, 
                                        before = pa.before, 
                                        after = pa.after)
        
        plos_scrapper_query_csv(filename = pa.query_csv_file, 
                                        lm = pa.lm, 
                                        m = pa.m, 
                                        d_range = d_range, 
                                        before = pa.before, 
                                        after = pa.after)

        time.sleep(2)
        runpy.run_path("text_analysis_v4.py")
        return

    elif (pa.query_param is not None):
        gnews_scrapper_query(lm = pa.lm, m = pa.m, d_range = d_range, before = pa.before, after = pa.after, q = pa.query_param)
        plos_scrapper_query(lm = pa.lm, m = pa.m, d_range = d_range, before = pa.before, after = pa.after, q = pa.query_param)

        time.sleep(2)
        runpy.run_path("text_analysis_v4.py")
        return

    else:
        print("Search query required.")
        return

    return

## Execute main
if __name__ == "__main__":
    main()