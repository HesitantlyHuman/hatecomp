import os
import shutil

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