import os
import csv
import shutil

from hatecomp.base.download import _ZipDownloader

class HASOCDownloader(_ZipDownloader):
    DOWNLOAD_URL = 'https://hasocfire.github.io/hasoc/2019/files/english_dataset.zip'
    def __init__(self, save_path, unzip=True, keep_zip=False, chunk_size=128) -> None:
        url = self.DOWNLOAD_URL
        super().__init__(url, save_path, unzip=unzip, keep_zip=keep_zip, chunk_size=chunk_size)

    def cleanup(self, path: str) -> None:
        with open(os.path.join(path, 'english_dataset/english_dataset.tsv'), 'r') as train_csv_file:
            train_csv = list(csv.reader(train_csv_file, delimiter = '\t'))
        with open(os.path.join(path, 'english_dataset/hasoc2019_en_test-2919.tsv'), 'r') as test_csv_file:
            test_csv = list(csv.reader(test_csv_file, delimiter = '\t'))
        hasoc_csv_data = []
        for item in train_csv[1:]:
            item = ['HASOC-train' + item[0].split('_')[-1]] + item[1:]
            hasoc_csv_data.append(item)
        for item in test_csv[1:]:
            item = ['HASOC-test' + item[0].split('_')[-1]] + item[1:]
            hasoc_csv_data.append(item)
        with open(os.path.join(path, 'hasoc.csv'), 'w') as HASOC_file:
            writer = csv.writer(HASOC_file)
            keys = ['id', 'data'] + train_csv[0][2:]
            writer.writerow(keys)
            for item in hasoc_csv_data:
                writer.writerow(item)
        paths_to_delete = [
            os.path.join(path, '__MACOSX'),
            os.path.join(path, 'english_dataset')
        ]
        for rm_path in paths_to_delete:
            shutil.rmtree(rm_path)