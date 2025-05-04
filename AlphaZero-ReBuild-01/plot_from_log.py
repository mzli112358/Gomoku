import os
import re
import csv
import matplotlib.pyplot as plt

log_path = 'train_log.txt'
csv_path = 'train_metrics_from_log.csv'

def parse_log_to_csv(log_path, csv_path):
    pattern = re.compile(r'kl:(?P<kl>[\d.]+), lr_multiplier:(?P<lr>[\d.]+), loss:(?P<loss>[\d.]+), entropy:(?P<entropy>[\d.]+)')
    batch_num = 0
    with open(log_path, 'r', encoding='utf-8') as fin, open(csv_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['batch','kl','loss','entropy'])
        for line in fin:
            if '训练批次:' in line:
                batch_num = int(re.findall(r'训练批次:\s*(\d+)', line)[0])
            m = pattern.search(line)
            if m:
                writer.writerow([batch_num, m.group('kl'), m.group('loss'), m.group('entropy')])
    print(f"解析完毕，保存到 {csv_path}")

def plot_metrics(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    plt.plot(df['batch'], df['loss'], label='Loss')
    plt.plot(df['batch'], df['entropy'], label='Entropy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parse_log_to_csv(log_path, csv_path)
    plot_metrics(csv_path)