import os
import shutil

from hatecomp.base.download import _ZipDownloader
from hatecomp._path import install_path

class HASOCDownloader(_ZipDownloader):
    DEFAULT_DIRECTORY = os.path.join(install_path, 'datasets/HASOC/data')
    DOWNLOAD_URL = 'https://hasocfire.github.io/hasoc/2019/files/english_dataset.zip'
    def __init__(self, save_path = None, unzip=True, keep_zip=False, chunk_size=128) -> None:
        if save_path is None:
            save_path = self.DEFAULT_DIRECTORY
        url = self.DOWNLOAD_URL
        super().__init__(url, save_path, unzip=unzip, keep_zip=keep_zip, chunk_size=chunk_size)