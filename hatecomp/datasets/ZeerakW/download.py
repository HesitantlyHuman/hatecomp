import os
import csv

from hatecomp.twitter.fetch import TweetDownloader
from hatecomp.base.download import _CSVDownloader
from hatecomp._path import install_path

#Split into two downloaders
class ZeerakWDownloader(_CSVDownloader):
    DEFAULT_DIRECTORY = os.path.join(install_path, 'datasets/ZeerakW/data')
    DOWNLOAD_URLs = [
        'https://raw.githubusercontent.com/zeeraktalat/hatespeech/master/NAACL_SRW_2016.csv',
        'https://raw.githubusercontent.com/zeeraktalat/hatespeech/master/NLP%2BCSS_2016.csv'
    ]

    def __init__(self, save_path = None) -> None:
        if save_path is None:
            save_path = self.DEFAULT_DIRECTORY
        urls = self.DOWNLOAD_URLs
        super().__init__(urls, save_path)

    def cleanup(self) -> None:
        self._cleanup_NAACL()
        self._cleanup_NLPCSS()

    def _cleanup_NLPCSS(self) -> None:
        with open(self.SAVE_PATHS[self.DOWNLOAD_URLs[1]], 'r') as NLPCSS_file:
            NLPCSS_csv = list(csv.reader(NLPCSS_file, delimiter = '\t'))
        ids = [row[0] for row in NLPCSS_csv[1:]]
        with TweetDownloader() as tweet_downloader:
            tweets = tweet_downloader.download(ids)
        updated_NLPCSS_csv = []
        for item, tweet in zip(NLPCSS_csv[1:], tweets):
            item = [
                'ZeerakW_NLPCSS-' + item[0],
                tweet,
                *item[2:]
            ]
            updated_NLPCSS_csv.append(item)
        with open(self.SAVE_PATHS[self.DOWNLOAD_URLs[1]], 'w') as NLPCSS_file:
            writer = csv.writer(NLPCSS_file)
            keys = ['id', 'data']
            for label in NLPCSS_csv[0][2:]:
                keys.append(label.lower())
            writer.writerow(keys)
            for item in updated_NLPCSS_csv:
                writer.writerow(item)

    def _cleanup_NAACL(self) -> None:
        with open(self.SAVE_PATHS[self.DOWNLOAD_URLs[0]], 'r') as NAACL_file:
            NAACL_csv = list(csv.reader(NAACL_file))
        ids = [item[0] for item in NAACL_csv]
        with TweetDownloader() as tweet_downloader:
            tweets = tweet_downloader.download(ids)
        updated_NAACL_csv = []
        for item, tweet in zip(NAACL_csv, tweets):
            item = [
                'ZeerakW_NAACL-' + item[0],
                tweet,
                item[1]
            ]
            updated_NAACL_csv.append(item)
        with open(self.SAVE_PATHS[self.DOWNLOAD_URLs[0]], 'w') as NAACL_file:
            writer = csv.writer(NAACL_file)
            writer.writerow(['id', 'data', 'hate_type'])
            for item in updated_NAACL_csv:
                writer.writerow(item)