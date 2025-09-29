from Focused_Crawler import crawl_website
from Web_Scraper import web_scraper
from EDA import clean_data
import time

# Get Website Link
print('_'*40)
web_site = input('Enter Website to Crawl and Collect Data : ')
over_all_time = time.time()

# Crawl and get Links
skip_patterns = ['/ar', '/upload', '/video']
max_pages = 1000
links_csv  = crawl_website(web_site, max_pages=max_pages, skip_patterns=skip_patterns)

# Scrap Data from collected Links
scraped_data_csv = web_scraper(links_csv)

# Clean scraped Data
cleaned_csv = clean_data(scraped_data_csv)
over_all_time = time.time() - over_all_time
print(f'[INFO] Data collected and stored in : {cleaned_csv}, Time taken : {over_all_time:.2f}')
