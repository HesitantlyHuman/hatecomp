import os

from hatecomp.base.download import _ZipDownloader
from hatecomp._path import install_path

class MLMADownloader(_ZipDownloader):
    DEFAULT_DIRECTORY = os.path.join(install_path, 'datasets/MLMA/data')
    DOWNLOAD_URL = 'https://github.com/HKUST-KnowComp/MLMA_hate_speech/raw/master/hate_speech_mlma.zip'
    def __init__(self, save_path = None, unzip=True, keep_zip=False, chunk_size=128) -> None:
        if save_path is None:
            save_path = self.DEFAULT_DIRECTORY
        url = self.DOWNLOAD_URL
        super().__init__(url, save_path, unzip=unzip, keep_zip=keep_zip, chunk_size=chunk_size)