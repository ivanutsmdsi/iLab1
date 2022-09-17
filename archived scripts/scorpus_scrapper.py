# pip install pyscopus==1.0.3a2

from pyscopus import Scopus

key = '2eac42012dc5ec0b92f9d9e1f714e0e2'        ## Ivan's Key - Linked to UTS Student Account (may expire after graduation)

scopus = Scopus(key)

search_df = scopus.search("KEY(interdisciplinary collaboration)", count=20, view='STANDARD')

print(search_df)

#pub_info = scopus.retrieve_abstract('85121580402', view='META_ABS')

#print(pub_info.keys())

#print(pub_info['abstract'])