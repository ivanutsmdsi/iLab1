import requests
import csv
import traceback
import random
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

URL_ori = "https://pubmed.ncbi.nlm.nih.gov/?term=Construction+Health+Safety"
headers = requests.utils.default_headers()
headers.update({
    'User-Agent': 'Mozilla/15.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20210916 Firefox/95.0',
})

try:

    # SETTING UP THE CSV FILE

    outfile = open("pubmed_data.csv","w",newline='',encoding='utf-8')
    writer = csv.writer(outfile)
    df = pd.DataFrame(columns=['Title','Journal','Abstract'])

    # SETTING & GETTING PAGE NUMBER
    page_num = 1
    page_view = 20 # can be change to 10, 20, 50, 100 or 200
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
    i = i +1
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
        abstract_element = job_element.find("div", class_="abstract-content selected")

        title_element_clean = title_element.a.text.strip()
        journal_element_clean = journal_element['title']
        abstract_element_clean = abstract_element.get_text().strip().replace('\n',' ').replace('\t',' ')
        
        ## test print
        # print(title_element_clean)
        # print(journal_element_clean)
        # print(abstract_element_clean)
        
        print("saving article")

        # exit()      # stop code for testing

        df2 = pd.DataFrame([[title_element_clean, journal_element_clean, abstract_element_clean]],columns=['Title','Journal','Abstract'])
        df = pd.concat([df, df2], ignore_index=True)

    wait()

except AttributeError:
    traceback.print_exc()
    print("Opss! ReCaptcha is probably preventing the code from running.")
    print("Please consider running in another time.\n")
    exit()

df.index += 1
df.to_csv('pubmed_data.csv')
outfile.close()
print("all saved")