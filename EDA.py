import pandas as pd
import re
import time

# Different Data clean up functions

def remove_arabic(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    text = arabic_pattern.sub('', text)
    return text


def remove_pics(text):
    text = re.sub(r'\S*\.png\S*', '', text)
    text = re.sub(r'\S*\.jpg\S*', '', text)
    return re.sub(r'\S*\.jpeg\S*', '', text)


def remove_links(text):
    return re.sub(r'\S*http\S*', '', text)


def remove_spaces(text):
    return re.sub(r'\s+', ' ', text)



def remove_short(text):
    # print(len(text))
    return text if len(text) > 50 else ''


def remove_dots(text):
    return text if '...' not in text else ''


def remove_repeated(text,counting_dict,max_dup):
    if text not in counting_dict:
        return ''
    return text if counting_dict[text] < max_dup else ''


def lowercase(text):
    return text.lower()


def clean_text(text,counting_dict,max_dup):
    text = remove_arabic(text)
    text = remove_links(text)
    text = remove_pics(text)
    text = remove_repeated(text,counting_dict,max_dup)

    text = lowercase(text)

    text = remove_spaces(text)

    return text.strip()

# To be used to remove duplicate data
def add_count(text,counting_dict):
    # global counting_dict

    if text in counting_dict:
        counting_dict[text] += 1
    else:
        counting_dict[text] = 1

def clean_data(csv_file):
    # Reading csv data
    start_time = time.time()
    print('_' * 40)
    print(f'Preforming EDA on {csv_file}')
    df = pd.read_csv(csv_file, index_col=0)

    # Initializations to be used

    texts = df['Text'].to_list()
    num_texts = len(texts)
    max_dup = 3
    counting_dict = {}

    for text in texts:
        for line in str(text).splitlines():
            add_count(line,counting_dict)

    counter = len(texts)

    # EDA on all data in the csv

    new_list = []
    for i in range(counter):
        new_line = []
        text = str(texts[i])
        for line in text.splitlines():
            line = clean_text(line,counting_dict,max_dup)

            # Remove empty lines
            if line:
                new_line.append(line)
        new_text = ''
        if len(new_line) > 1:
            new_text = ' '.join(new_line)
        new_list.append(new_text)

    # Saving the cleaned data

    df['Text'] = new_list
    output_file = csv_file[:-4] + '_eda.csv'
    df.to_csv(output_file, index=False)
    end_time = time.time() - start_time
    print(f'[INFO] Finished Script. Time Take {end_time:.2f}s')
    print('_' * 40)

    return output_file