import os

from hatecomp.download import ZipDownloader

DEFAULT_SAVE_LOCATION = 'hatecomp/datasets/HASOC/data'
DOWNLOAD_URL = 'https://hasocfire.github.io/hasoc/2019/files/english_dataset.zip'
PATHS_TO_CLEANUP = [
    'hatecomp/datasets/data/__MACOSX'
]

class HASOCDownloader(ZipDownloader):
    def __init__(self, save_path = None, unzip=True, keep_zip=False, chunk_size=128) -> None:
        if save_path is None:
            save_path = DEFAULT_SAVE_LOCATION
        url = DOWNLOAD_URL
        super().__init__(url, save_path, unzip=unzip, keep_zip=keep_zip, chunk_size=chunk_size)

    def cleanup(self, path: str) -> None:
        super().cleanup(path)
        for path in PATHS_TO_CLEANUP:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                os.rmdir(path)

if __name__ == '__main__':
    downloader = HASOCDownloader()
    downloader.load()