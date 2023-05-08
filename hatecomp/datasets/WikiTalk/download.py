import csv
import os

from hatecomp.datasets.base.download import download_files


class _WikiDownloader:
    DOWNLOAD_URLs = None

    def __init__(self, save_path: str) -> None:
        self.urls = self.DOWNLOAD_URLs
        self.save_path = save_path

    def load(self) -> None:
        print(f"{type(self).__name__} Downloading {len(self.urls)} files...")
        urls = self.urls.values()
        download_files(urls, self.save_path)
        self.cleanup()

    def cleanup(self) -> None:
        # Chane the name of the files to something more readable
        for file_name, url in self.urls.items():
            old_path = os.path.join(self.save_path, url.split("/")[-1])
            new_path = os.path.join(self.save_path, file_name)
            os.rename(old_path, new_path)


class WikiToxicityDownloader(_WikiDownloader):
    DOWNLOAD_URLs = {
        "toxicity_annotations.tsv": "https://figshare.com/ndownloader/files/7394539",
        "toxicity_annotated_comments.tsv": "https://figshare.com/ndownloader/files/7394542",
    }


class WikiAggressionDownloader(_WikiDownloader):
    DOWNLOAD_URLs = {
        "aggression_annotations.tsv": "https://figshare.com/ndownloader/files/7394506",
        "aggression_annotated_comments.tsv": "https://figshare.com/ndownloader/files/7038038",
    }


class WikiPersonalAttacksDownloader(_WikiDownloader):
    DOWNLOAD_URLs = {
        "attack_annotations.tsv": "https://figshare.com/ndownloader/files/7554637",
        "attack_annotated_comments.tsv": "https://figshare.com/ndownloader/files/7554634",
    }


if __name__ == "__main__":
    downloader = WikiToxicityDownloader("hatecomp/datasets/WikiTalk/data")
    downloader.load()

    downloader = WikiAggressionDownloader("hatecomp/datasets/WikiTalk/data")
    downloader.load()

    downloader = WikiPersonalAttacksDownloader("hatecomp/datasets/WikiTalk/data")
    downloader.load()
