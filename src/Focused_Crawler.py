import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import pandas as pd
import re
from pathlib import Path


# Crawl through urls of a base_url in BFS traversal, avoiding any urls that have a skip_pattern

def crawl_website(base_url, max_pages=100, skip_patterns=None):
    print('_'*40)
    print(f'[INFO] Crawling Started : {base_url}')

    # Initializations
    start_time = time.time()
    visited = set()
    queue = deque([base_url])
    domain = urlparse(base_url).netloc
    results = []
    count = 0
    skip_patterns = skip_patterns or []


    # BFS Traversal on urls found
    while queue and len(visited) < max_pages:
        url = queue.popleft()

        if url in visited:
            continue

        try:
            # Process url
            response = requests.get(url)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            visited.add(url)
            title = soup.title.string.strip() if soup.title else 'No Title'
            results.append((title, url))
            count += 1

            print(f'[INFO] page# {count} {title}: {url}')

            # Find connected urls
            for a in soup.find_all('a', href=True):
                link = urljoin(base_url, a['href'])

                if urlparse(link).netloc == domain and link not in visited:
                    if not any(p in link for p in skip_patterns):
                        queue.append(link)
        except Exception as e:
            print(f'[ERROR] {url} : {e}')
            continue

    end_time = time.time() - start_time
    print(f'[INFO] Finished Script. Time Take {end_time:.2f}s')
    print('_'*40)

    # Get names and urls to save
    names = [x[0] for x in results]
    urls = [x[1] for x in results]
    df = pd.DataFrame({'Name': names,
                       'URL': urls})

    # Saved the Scrapped links in a .csv file using Pandas
    ROOT = Path(__file__).resolve().parent.parent
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    match = re.search(r"https?://(?:www\.)?([^./]+)", base_url)
    web_site_name = match.group(1)
    output_file = web_site_name + '_crawled_links.csv'
    output_file = data_dir / output_file
    df.to_csv(output_file, index=False)

    # Return csv file
    return output_file

