import os
import csv

from hatecomp.tweets.fetch import TweetDownloader
from hatecomp.base.download import _CSVDownloader


class TwitterSexismDownloader(_CSVDownloader):
    DOWNLOAD_URLs = [
        "https://raw.githubusercontent.com/AkshitaJha/NLP_CSS_2017/master/benevolent_sexist.tsv",
        "https://raw.githubusercontent.com/AkshitaJha/NLP_CSS_2017/master/hostile_sexist.tsv",
    ]

    def __init__(self, save_path: str) -> None:
        urls = self.DOWNLOAD_URLs
        super().__init__(urls, save_path)

    def cleanup(self) -> None:
        with open(self.SAVE_PATHS[self.DOWNLOAD_URLs[0]], "r") as id_file:
            benevolent_id_list = [row[0] for row in csv.reader(id_file)]
        with open(self.SAVE_PATHS[self.DOWNLOAD_URLs[1]], "r") as id_file:
            hostile_id_list = [row[0] for row in csv.reader(id_file)]
        benevolent_id_list, hostile_id_list = list(set(benevolent_id_list)), list(
            set(hostile_id_list)
        )
        tweet_ids = benevolent_id_list + hostile_id_list
        with TweetDownloader() as tweet_downloader:
            tweets = tweet_downloader.download(tweet_ids)
        print(len(tweets))
        print(tweets.keys())
        twitter_sexism_csv = []
        for tweet_id in tweet_ids:
            if tweet_id in hostile_id_list:
                label = "hostile"
            elif tweet_id in benevolent_id_list:
                label = "benevolent"
            try:
                item = ["TwitterSexism-" + tweet_id, tweets[tweet_id], label]
                twitter_sexism_csv.append(item)
            except KeyError:
                pass
        with open(os.path.join(self.save_path, "twitter_sexism.csv"), "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["id", "data", "sexism_type"])
            for item in twitter_sexism_csv:
                writer.writerow(item)
        os.remove(os.path.join(self.save_path, "benevolent_sexist.tsv"))
        os.remove(os.path.join(self.save_path, "hostile_sexist.tsv"))
