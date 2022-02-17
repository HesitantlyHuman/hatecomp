import csv
from collections import Counter

from hatecomp.tweets.fetch import TweetDownloader
from hatecomp.base.download import _CSVDownloader


class NAACLDownloader(_CSVDownloader):
    DOWNLOAD_URL = "https://raw.githubusercontent.com/zeeraktalat/hatespeech/master/NAACL_SRW_2016.csv"

    def __init__(self, save_path: str) -> None:
        urls = [self.DOWNLOAD_URL]
        super().__init__(urls, save_path)

    def cleanup(self) -> None:
        with open(self.SAVE_PATHS[self.DOWNLOAD_URL], "r") as NAACL_file:
            NAACL_csv = list(csv.reader(NAACL_file))
        ids = [item[0] for item in NAACL_csv]
        with TweetDownloader() as tweet_downloader:
            tweets = tweet_downloader.download(ids)
        updated_NAACL_csv = []
        for item in NAACL_csv:
            try:
                item = ["ZeerakW_NAACL-" + item[0], tweets[item[0]], item[1]]
                updated_NAACL_csv.append(item)
            except KeyError:
                pass
        with open(self.SAVE_PATHS[self.DOWNLOAD_URL], "w") as NAACL_file:
            writer = csv.writer(NAACL_file)
            writer.writerow(["id", "data", "hate_type"])
            for item in updated_NAACL_csv:
                writer.writerow(item)


class NLPCSSDownloader(_CSVDownloader):
    DOWNLOAD_URL = "https://raw.githubusercontent.com/zeeraktalat/hatespeech/master/NLP%2BCSS_2016.csv"

    def __init__(self, save_path: str) -> None:
        urls = [self.DOWNLOAD_URL]
        super().__init__(urls, save_path)

    def cleanup(self) -> None:
        # Open the downloaded csv file
        with open(self.SAVE_PATHS[self.DOWNLOAD_URL], "r") as NLPCSS_file:
            NLPCSS_csv = list(csv.reader(NLPCSS_file, delimiter="\t"))

        # Find which indices correspond to the two labeler categories
        amateur_indices = []
        expert_indices = []
        for index, header in enumerate(NLPCSS_csv[0]):
            if "amateur" in header.lower():
                amateur_indices.append(index)
            if "expert" in header.lower():
                expert_indices.append(index)

        # Download tweet content
        ids = [row[0] for row in NLPCSS_csv[1:]]
        with TweetDownloader() as tweet_downloader:
            tweets = tweet_downloader.download(ids)

        # Create a new csv with the tweet content and pooled ratings
        updated_NLPCSS_csv = []
        for item in NLPCSS_csv[1:]:
            amateur_vote = NLPCSSDownloader.get_vote(item, amateur_indices)
            expert_vote = NLPCSSDownloader.get_vote(item, expert_indices)
            try:
                item = [
                    "ZeerakW_NLPCSS-" + item[0],
                    tweets[item[0]],
                    expert_vote,
                    amateur_vote,
                ]
                updated_NLPCSS_csv.append(item)
            except KeyError:
                pass

        # Save the new csv over the old one
        with open(self.SAVE_PATHS[self.DOWNLOAD_URL], "w") as NLPCSS_file:
            writer = csv.writer(NLPCSS_file)
            keys = ["id", "data", "expert", "amateur"]
            writer.writerow(keys)
            for item in updated_NLPCSS_csv:
                writer.writerow(item)

    def get_vote(item, indices):
        labels = [item[i] for i in indices]
        for value, _ in Counter(labels).most_common():
            if not value == "":
                return value
        return ""
