import pandas as pd
import numpy as np
import requests
import os
import csv
import argparse
# your settings
parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, default=None)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--token', type=str, default='songid')
parser.add_argument('--lyric_batch_size', type=int, default=100)
args = parser.parse_args()

csv_path = args.csv_file
output_file = args.output_file

writer = csv.writer(open(output_file, 'w'))
sep = ','

batch_size = args.lyric_batch_size

def main():
    df = pd.read_csv(csv_path)
    songids = df[args.token].astype(str).tolist()
#     tags = df.genre.astype(str).tolist()
    print('total {} samples to extract ...'.format(len(songids)))

    n_batch = int(len(songids)/batch_size) + 1
    n = 0
    zh_k = 0
    for i in range(n_batch):
        sub_songids = songids[i*batch_size:(i+1)*batch_size]
        resq_params = {'id': ', '.join(sub_songids), 'clean': 'deep'}
        resq = requests.post('http://11.181.92.137:8080/lyric_pull', json=resq_params)
        results = resq.json()
        for j in range(len(results)):
            if results[j]['lyric'] != '':
                line = [results[j]['id'], results[j]['lyric']]
                writer.writerow(line)
            else:
                pass
            n = n + 1
        print('finish {} samples ...'.format(n))



if __name__ == '__main__':
    main()
