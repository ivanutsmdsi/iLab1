import requests
import csv
import random
import traceback
import time
import pandas as pd
from sys import exit
from bs4 import BeautifulSoup


##
# web_scrap code adapted from 'ScrapPaper' by M. R. Rafsanjani, Completed on 2022 Feb 13th, Penang, Malaysia.
# Modifications made by iLab1 'Safety First' - MDSI Spring 2022
##
def wait():
    print("Waiting for a few secs...")
    time.sleep(random.randrange(1, 6))
    print("Waiting done. Continuing...\n")


# ===== GETTING AND SETTING THE URL =====

#########################################
# CHANGE LOG
# 21-08-2022 Added in 2 search variants
# 23-08-2022 Added publication date, filter year range.

# TO DO List - cited number code is not working yet.

#########################################
# William's attempt to use a list with fstrings to dynamically update the URL with keyword search terms
# Variant 1 - cycle through a list of keywords
# Variant 2 - Ask user through input to see what they would like to search

# ===== Variant 1 ======================
# import random
# below words are based on 'buckets' identified through research
# topic = "construction"
# bucket2 = ["health", "safety", "OHS", "accident"]
# bucket3 = ["stress", "absentee", "illness", "sick"]
# bucket4 = ["hazard", "injury", "risk", "joint", "prevent", "fatal"]

# word2 = random.choice(bucket2)
# word3 = random.choice(bucket3)
# word4 = random.choice(bucket4)

# final_search = f"{topic}+{word2}+{word3}+{word4}"

# ===== Variant 2 ======================
keyword1, keyword2, keyword3 = input("Enter 3 search terms separated by a space: ").split()
print("Keyword 1:", keyword1)
print("Keyword 2:", keyword2)
print("Keyword 3:", keyword3)

search_string = f"{keyword1}+{keyword2}+{keyword3}"

start_year, end_year = input("Please enter start and end year to filter search, separated by a space: ").split()
print("Start year:", start_year)
print("End year:", end_year)

filter_years = f"years.{start_year}-{end_year}"

# URL_ori = f"https://pubmed.ncbi.nlm.nih.gov/?term={final_search}" -- UNCOMMENT TO USE VARIANT 1
URL_ori = f"https://pubmed.ncbi.nlm.nih.gov/?term={search_string}&filter={filter_years}"
headers = requests.utils.default_headers()
headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36'
        #'Mozilla/15.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20210916 Firefox/95.0',
})

##### -- Attempt to use rotating list of User Agents to work around IP Blocking






# END OF MODIFICATIONS
########################################


try:

    # SETTING UP THE CSV FILE

    outfile = open("pubmed_data.csv", "w", newline='', encoding='utf-8')
    writer = csv.writer(outfile)
    df = pd.DataFrame(columns=['Title', 'Journal', 'Publication Date', 'Cited by','Abstract'])

    # SETTING & GETTING PAGE NUMBER
    page_num = 1
    page_view = 100  # can be change to 10, 20, 50, 100 or 200
    URL_edit = URL_ori + "&page=" + str(page_num) + "&size=" + str(page_view) + "&format=abstract"

    page = requests.get(URL_edit, headers=headers, timeout=None)
    soup = BeautifulSoup(page.content, "html.parser")
    wait()

    page_total = soup.find("label", class_="of-total-pages").text
    page_total_num = int(''.join(filter(str.isdigit, page_total)))
    print(f"Total page number: {page_total_num}")
    print(f"Results per page: {page_view}.\n")

except AttributeError:

    print("Opss! ReCaptcha is probably preventing the code from running.")
    print("Please consider running in another time.\n")
    exit()

wait()

# EXTRACTING INFORMATION

# for i in range(page_total_num):       ## iterates through all pages


i = 0
while i < 1:
    i = i + 1
    page_num_up = page_num + i
    URL_edit = URL_ori + "&page=" + str(page_num_up) + "&size=" + str(page_view) + "&format=abstract"
    print("URL : ", URL_edit)
    page = requests.get(URL_edit, headers=headers, timeout=None)

    soup = BeautifulSoup(page.content, "html.parser")
    wait()
    print('loading results')
    results = soup.find("section", class_="search-results-list")

try:

    # EXTRACTING INFORMATION
    # print(results)	
    print('finding articles')
    job_elements = results.find_all("article", class_="article-overview")
    # print(job_elements)

    for job_element in job_elements:
        title_element = job_element.find("h1", class_="heading-title")
        journal_element = job_element.find("button", class_="journal-actions-trigger trigger")
        pubdate_element = job_element.find("span", class_="cit")
        # citation_element = job_element.find("li", class_="citedby-count")
        abstract_element = job_element.find("div", class_="abstract-content selected")

        title_element_clean = title_element.a.text.strip()
        journal_element_clean = journal_element['title']
        sep = ';'
        pubdate_element_clean = pubdate_element.get_text().split(sep, 1)[0]
        # citation_element_clean = citation_element.split()[-2]
        abstract_element_clean = abstract_element.get_text().strip().replace('\n', ' ').replace('\t', ' ')

        ## test print
        # print(title_element_clean)
        # print(journal_element_clean)
        # print(pubdate_element_clean)
        # print(citation_element_clean)
        # print(abstract_element_clean)

        print("saving article")

        # exit()      # stop code for testing

        df2 = pd.DataFrame([[title_element_clean,
                             journal_element_clean,
                             pubdate_element_clean,
                             # citation_element_clean,
                             abstract_element_clean]],
                           columns=['Title', 'Journal', 'Publication Date', 'Abstract'])
                            # columns = ['Title', 'Journal', 'Publication Date', 'Cited by', 'Abstract'])
        df = pd.concat([df, df2], ignore_index=True)

    wait()

except AttributeError:
    traceback.print_exc()
    print("Oops! Recaptcha is probably preventing the code from running.")
    print("Please consider running in another time.\n")
    exit()

df.index += 1
df.to_csv('pubmed_data.csv')
outfile.close()
print("all saved")
