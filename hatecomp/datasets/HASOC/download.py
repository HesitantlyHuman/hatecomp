import os
import shutil

from hatecomp.download import ZipDownloader
from hatecomp._path import install_path

DEFAULT_DIRECTORY = os.path.join(install_path, 'datasets/HASOC/data')
DOWNLOAD_URL = 'https://hasocfire.github.io/hasoc/2019/files/english_dataset.zip'
PATHS_TO_CLEANUP = [
    '/datasets/data/__MACOSX'
]

class HASOCDownloader(ZipDownloader):
    def __init__(self, save_path = None, unzip=True, keep_zip=False, chunk_size=128) -> None:
        if save_path is None:
            save_path = DEFAULT_DIRECTORY
        url = DOWNLOAD_URL
        super().__init__(url, save_path, unzip=unzip, keep_zip=keep_zip, chunk_size=chunk_size)

    #Currently not working, removed to limit side effects
    def cleanup(self, path: str) -> None:
        super().cleanup(path)
        '''
        for path in PATHS_TO_CLEANUP:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        '''