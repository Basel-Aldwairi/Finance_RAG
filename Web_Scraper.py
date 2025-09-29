import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import re

# Goes through the urls in the csv file and scrapes the HTML

def web_scraper(csv_input_file):
    print('_' * 40)
    print(f'[INFO] Scraping Started on urls from : {csv_input_file}')

    # Read csv
    df = pd.read_csv(csv_input_file)

    # Ready the session for scraping
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive"
    }

    session = requests.Session()
    session.headers.update(headers)

    bank_time = time.time()

    names, urls = df['Name'], df['URL']
    size = len(names)
    df['Text'] = pd.NA
    # Get name of website
    match = re.search(r"https?://(?:www\.)?([^./]+)", str(urls[0]))
    bank = match.group(1)

    # oop through all the urls and scrap the data
    for i in range(size):
        url = urls[i]
        name = names[i]
        print(f'[ INFO: {i + 1}/{size}] Visiting {name} at {url}')

        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')

            for tag in soup(['header', 'footer', 'nav', 'aside', 'script', 'style']):
                tag.decompose()

            text = soup.get_text(separator='\n')
            text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])

            # Filter very short data
            if len(text) >= 1000:
                df.at[i, 'Text'] = text

            print(f'[INFO: {i + 1}/{size}] Scraped {len(text)} from {url}')

        except Exception as e:
            print(f'[INFO: {i + 1}/{size}] Error scraping {url}: {e}')

    # Filter out Empty urls
    df = df.dropna(subset=['Text'])
    output_file = f'{bank}_scraped_data.csv'

    # Save data in a new file
    df.to_csv(output_file)
    bank_end = time.time() - bank_time
    print(f'[INFO] Saved Data in {output_file}')
    print(f'[INFO] Finished Script. Time Take {bank_end:.2f}s')
    print('_' * 40)

    # Return the csv file of the data
    return output_file
