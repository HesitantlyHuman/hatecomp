import os
import shutil
import csv

from hatecomp.twitter.fetch import TweetDownloader
from hatecomp.base.download import _CSVDownloader
from hatecomp._path import install_path

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

    def load(self) -> None:
        super().load()
        self.get_tweets()

    def get_tweets(self) -> None:
        with open(self.SAVE_PATHS[self.DOWNLOAD_URLs[0]], 'r') as NAACL_file:
            NAACL_csv = list(csv.reader(NAACL_file))
        ids = [item[0] for item in NAACL_csv]
        tweets = ['hehe, haha, hehe, haha' for id in ids]
        with TweetDownloader() as tweet_downloader:
            pass
            #tweets = tweet_downloader.download(ids)
        NAACL_csv = [item.append(tweet) for item, tweet in zip(NAACL_csv, tweets)]
        print(NAACL_csv)
        with open(self.SAVE_PATHS[self.DOWNLOAD_URLs[0]], 'w') as NAACL_file:
            writer = csv.writer(NAACL_file)
            for item in NAACL_csv:
                writer.writerow(item)