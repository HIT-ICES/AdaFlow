import os
import csv
import pandas as pd
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai
import time
import numpy as np
openai.api_key = "********************************"
base_path = 'Correted-ProgrammableWeb-dataset-main/data/'
processing_data_path = 'DS1/ada-001/processing_data/'
output_data_path = 'DS1/ada-001/output_data/'


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-001"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def generate_api_dict():
    active_api_data_path = os.path.join(base_path, 'raw/api_mashup/active_apis_data.txt')
    deadpool_api_data_path = os.path.join(base_path, 'raw/api_mashup/deadpool_apis_data.txt')
    f1 = open(active_api_data_path)
    active_api_list = json.load(f1)
    f2 = open(deadpool_api_data_path)
    deadpool_api_list = json.load(f2)
    api_dict = {}
    num_api = 0
    for api in active_api_list:
        if api is None:
            continue
        # url = api['url']
        url = api['url'][0:-1]
        if url not in api_dict:
            api_dict[url] = {
                'id': num_api,
                'des': api['description'],
                'title': api['title'],
                'tags': api['tags']
            }
            num_api += 1
    for api in deadpool_api_list:
        if api is None:
            continue
        # url = api['url']
        url = api['url'][0:-1]
        if url not in api_dict:
            api_dict[url] = {
                'id': num_api,
                'des': api['description'],
                'title': api['title'],
                'tags': api['tags']
            }
            num_api += 1
    print("in raw, num_api = ", num_api)
    df = json.dumps(api_dict)
    # save_path = os.path.join(processing_data_path, 'api_dict.json')
    # with open(save_path, 'w') as f:
    #     f.write(df)
    return api_dict

def generate_mashup_dict():
    active_mashup_data_path = os.path.join(base_path, 'raw/api_mashup/active_mashups_data.txt')
    deadpool_mashup_data_path = os.path.join(base_path, 'raw/api_mashup/deadpool_mashups_data.txt')
    f1 = open(active_mashup_data_path)
    active_mashup_list = json.load(f1)
    f2 = open(deadpool_mashup_data_path)
    deadpool_mashup_list = json.load(f2)

    mashup_dict = {}
    num_mashup = 0
    for mashup in active_mashup_list:
        if mashup is None:
            continue
        title = mashup['title']
        if title not in mashup_dict and len(mashup['description']) != 0:
            count = 0
            for api in mashup['related_apis']:
                if api is None or api['url'] is None or len(api['description']) == 0:
                    continue
                count += 1
            if count <= 0:
                continue
            date = mashup['date'][0:-1] if mashup['date'][-1] == '\n' else mashup['date']
            mashup_dict[title] = {
                'id': num_mashup,
                'des': mashup['description'],
                'mashup_type': mashup['mashup_type'],
                'categories': mashup['categories'],
                'date': date,
                'url': mashup['url'],
                'related_apis': []
            }
            for api in mashup['related_apis']:
                if api is None or api['url'] is None or len(api['description']) == 0:
                    continue
                mashup_dict[title]['related_apis'].append({
                    'url': api['url'],
                    'des': api['description'],
                    'title': api['title'],
                    'tags': api['tags']
                })
            num_mashup += 1
    for mashup in deadpool_mashup_list:
        if mashup is None:
            continue
        title = mashup['title']
        if title not in mashup_dict and len(mashup['description']) != 0:
            count = 0
            for api in mashup['related_apis']:
                if api is None or api['url'] is None or len(api['description']) == 0:
                    continue
                count += 1
            if count <= 0:
                continue
            date = mashup['date'][0:-1] if mashup['date'][-1] == '\n' else mashup['date']
            mashup_dict[title] = {
                'id': num_mashup,
                'des': mashup['description'],
                'mashup_type': mashup['mashup_type'],
                'categories': mashup['categories'],
                'date': date,
                'url': mashup['url'],
                'related_apis': []
            }
            for api in mashup['related_apis']:
                if api is None or api['url'] is None or len(api['description']) == 0:
                    continue
                mashup_dict[title]['related_apis'].append({
                    'url': api['url'],
                    'des': api['description'],
                    'title': api['title'],
                    'tags': api['tags']
                })
            num_mashup += 1
    df = json.dumps(mashup_dict)
    # save_path = os.path.join(processing_data_path, 'mashup_dict.json')
    # with open(save_path, 'w') as f:
    #     f.write(df)
    return mashup_dict

def generate_data_onlyusemashup():
    if not os.path.exists(os.path.join(processing_data_path, 'mashup_dict.json')):
        mashup_dict = generate_mashup_dict()
    else:
        with open(os.path.join(processing_data_path, 'mashup_dict.json')) as f:
            mashup_dict = json.load(f)

    # api mashup info
    api_file = open(os.path.join(output_data_path, 'api_info.csv'), 'w', encoding='utf-8', newline="")
    api_csv = csv.writer(api_file)
    mashup_file = open(os.path.join(output_data_path, 'mashup_info.csv'), 'w', encoding='utf-8', newline="")
    mashup_csv = csv.writer(mashup_file)
    mashup_csv.writerow(['id', 'url', 'title', 'description', 'mashup_type', 'categories', 'date'])
    api_csv.writerow(['id', 'url', 'title', 'description', 'tags'])

    # embeding data
    api_dev_embeds = []
    mashup_dev_embeds = []
    tag_embeds = []

    processing_invocation = {
        'time': [],
        'Xs': [],
        'Ys': []
    }
    mashup_col = {}
    tag_col = {}
    api_col = {}
    num_api = 0
    num_mashup = 0
    num_tag = 0
    for mashup_name in mashup_dict:
        mashup = mashup_dict[mashup_name]
        y = []
        for api in mashup['related_apis']:
            api_url = api['url']
            if api_url not in api_col:
                api_col[api_url] = num_api
                num_api += 1
                # api_dev_embeds.npz
                print("get_embeding for api des!")
                print(len(api['des']))
                api_dev_embeds.append(get_embedding(api['des']))
                # api_info.csv
                api_csv.writerow([
                    api_col[api_url], api_url, api['title'], api['des'], ' '.join(api['tags'])
                ])
                for tag in api['tags']:
                    if tag not in tag_col:
                        tag_col[tag] = num_tag
                        num_tag += 1
                        # print("get_embeding for api tag!")
                        tag_embeds.append(get_embedding(tag))
            y.append(api_col[api_url])
        if mashup_name not in mashup_col:
            mashup_col[mashup_name] = num_mashup
            num_mashup += 1
            # invocation.json
            processing_invocation['Xs'].append(mashup_col[mashup_name])
            processing_invocation['Ys'].append(y)
            processing_invocation['time'].append(mashup['date'])
            # mashup_dev_embed.npz
            print("get_embeding for mashup des!")
            mashup_dev_embeds.append(get_embedding(mashup['des']))
            # mashup_info.csv
            mashup_csv.writerow([
                mashup_col[mashup_name], mashup['url'], mashup_name, mashup['des'], mashup['mashup_type'],
                ' '.join(mashup['categories']), mashup['date']
            ])

    # api_dev_embeds.npz
    api_dev_embeds = np.array(api_dev_embeds)
    np.savez(os.path.join(output_data_path, 'api_dev_embeds.npz'), api_dev_embeds)
    # tag_embeds.npz
    tag_embeds = np.array(tag_embeds)
    np.savez(os.path.join(output_data_path, 'tag_embeds.npz'), tag_embeds)
    # mashup_dev_embed.npz
    mashup_dev_embeds = np.array(mashup_dev_embeds)
    np.savez(os.path.join(output_data_path, 'mashup_dev_embeds.npz'), mashup_dev_embeds)
    #
    df = json.dumps(processing_invocation)
    with open(os.path.join(processing_data_path, 'processing_invocation.json'), 'w') as f:
        f.write(df)
    # invocation.pkl
    df = pd.DataFrame(processing_invocation)
    df.to_pickle(os.path.join(processing_data_path, 'processing_invocation.pkl'))
    # analyis_invocation(processing_invocation)

def read_invocation():
    df = pd.read_pickle(os.path.join(processing_data_path, 'processing_invocation.pkl'))  # pickle中存储python字典对象的序列化
    df['time'] = pd.to_datetime(df['time'], format="%m.%d.%Y")  # 时间
    df.sort_values(by='time', inplace=True)  # 按时间排序
    df['time'] = (df['time'] - df['time'].min()).dt.days
    invocation = {
        'Xs': df.Xs.tolist(),
        'Ys': df.Ys.tolist(),
        'time': df.time.tolist()
    }
    length = len(df.Xs)
    train_len = int(length * 0.7)
    val_len = int(length * 0.1)
    test_len = length - train_len - val_len
    print(length, train_len, val_len, test_len)
    mask = [0] * length
    for i in range(train_len, train_len + val_len):
        mask[i] = 1
    for i in range(train_len + val_len, length):
        mask[i] = 2
    invocation['mask'] = mask
    df = json.dumps(invocation)
    with open(os.path.join(output_data_path, 'invocation.json'), 'w') as f:
        f.write(df)

def analyis():
    # 有多少api大概，有多少个mashup大概
    if not os.path.exists(os.path.join(processing_data_path, 'api_dict.json')):
        api_dict = generate_api_dict()
    else:
        with open(os.path.join(processing_data_path, 'api_dict.json')) as f:
            api_dict = json.load(f)
    if not os.path.exists(os.path.join(processing_data_path, 'mashup_dict.json')):
        mashup_dict = generate_mashup_dict()
    else:
        with open(os.path.join(processing_data_path, 'mashup_dict.json')) as f:
            mashup_dict = json.load(f)

    m_a_edges_reader = pd.read_csv(os.path.join(base_path, 'm-a_edges.csv'), encoding="utf-8",
                                   names=['source', 'target'], sep='\t')

    # api_collection mashup_collection
    api_url_collection = {}
    mashup_name_collection = {}
    num_api = 0
    num_tag = 0
    tag_embeds_mask = {}
    num_api_in_edges = 0
    num_mashup_in_edges = 0
    api_col = {}
    mashup_col = {}
    for index, row in m_a_edges_reader.iterrows():
        if index == 0:
            continue
        api_url = row['target']
        mashup_name = row['source']
        if api_url not in api_col:
            api_col[api_url] = num_api_in_edges
            num_api_in_edges += 1
        if api_url not in api_url_collection:
            if api_url in api_dict and mashup_name in mashup_dict:
                api_url_collection[api_url] = num_api
                num_api += 1
                for tag in api_dict[api_url]['tags']:
                    if tag in tag_embeds_mask:
                        continue
                    tag_embeds_mask[tag] = tag
                    num_tag += 1
    print("num_tag: ", num_tag)
    print("num_api: ", num_api)

    invocation_dict = {}
    num_mashup = 0
    total = 0
    for index, row in m_a_edges_reader.iterrows():
        if index == 0:
            continue
        total += 1
        mashup_name = row['source']
        api_url = row['target']
        if mashup_name not in mashup_col:
            mashup_col[mashup_name] = num_mashup_in_edges
            num_mashup_in_edges += 1
        if mashup_name not in mashup_name_collection:
            if mashup_name in mashup_dict and api_url in api_url_collection:
                mashup_name_collection[mashup_name] = num_mashup
                if mashup_name not in invocation_dict:
                    invocation_dict[mashup_name] = {
                        'id': num_mashup,
                        'invokes': [],
                        'time': mashup_dict[mashup_name]['date']
                    }
                invocation_dict[mashup_name]['invokes'].append(api_url_collection[api_url])
                num_mashup += 1
        elif api_url in api_url_collection:
            invocation_dict[mashup_name]['invokes'].append(api_url_collection[api_url])
    processing_invocation = {
        'time': [],
        'Xs': [],
        'Ys': []
    }
    for mashup_name in invocation_dict:
        mashup = invocation_dict[mashup_name]
        processing_invocation['Xs'].append(mashup['id'])
        processing_invocation['Ys'].append(mashup['invokes'])
        processing_invocation['time'].append(mashup['time'])
    print("num_mashup: ", num_mashup)
    print("num_api_in_edges: ", num_api_in_edges)
    print("num_mashup_in_edges: ", num_mashup_in_edges)
    print("total_invokes: ", total)
    print("average_invokes: ", total / num_mashup_in_edges)
    print("--------------------------------------------")
    analyis_invocation(processing_invocation)

def analyis_invocation(result):
    Xs = result['Xs']
    Ys = result['Ys']
    counter = {}
    num_mashup = 0
    num_invoke_api = 0
    api_dict = {}
    num_api = 0
    for y in Ys:
        invokes = len(y)
        num_invoke_api += invokes
        if invokes not in counter:
            counter[invokes] = 0
        counter[invokes] += 1
        for invoke in y:
            if invoke not in api_dict:
                api_dict[invoke] = invoke
                num_api += 1
    num_one_invoke = counter[1]
    num_two_invoke = counter[2]
    num_three_invoke = counter[3]
    num_four_invoke = counter[4]
    for invoke in counter:
        num_mashup += counter[invoke]
        print(invoke, ': ', counter[invoke])
    print('num_api: ', num_api)
    print('num_mashup: ', num_mashup)
    print('invoke_arverage: ', num_invoke_api / num_mashup)
    print('invoke_num_1: ', num_one_invoke / num_mashup * 100, '%')
    print('invoke_num_2: ', num_two_invoke / num_mashup * 100, '%')
    print('invoke_num_3: ', num_three_invoke / num_mashup * 100, '%')
    print('invoke_num_4: ', num_four_invoke / num_mashup * 100, '%')

if __name__ == '__main__':
    generate_data_onlyusemashup()
    read_invocation()