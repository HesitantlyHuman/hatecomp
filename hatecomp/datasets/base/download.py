import requests
import zipfile
import os
from tqdm import tqdm

class _CSVDownloader():
    SAVE_PATHS = {}

    def __init__(self, urls, save_path):
        self.urls = urls
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path

    def load(self) -> None:
        for url in self.urls:
            save_path = os.path.join(self.save_path, url.split('/')[-1])
            self.download(url, save_path)
            self.SAVE_PATHS[url] = save_path
        self.cleanup()

    def download(self, url: str, path: str) -> None:
        response = requests.get(url = url)
        with open(path, 'w') as csv:
            csv.write(response.text)

    def cleanup(self) -> None:
        pass

class _ZipDownloader():
    def __init__(
        self,
        url,
        save_path,
        unzip = True,
        keep_zip = False,
        chunk_size = 128
    ) -> None:
        self.url = url
        self.chunk_size = chunk_size
        self.do_unzip = unzip
        self.keep_zip = keep_zip

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path

    def load(self) -> None:
        temp_path = os.path.join(self.save_path, 'temp.zip')
        self.download(url = self.url, path = temp_path, chunk_size = self.chunk_size)
        if self.do_unzip:
            self.unzip(origin_path = temp_path)
            if not self.keep_zip:
                os.remove(path = temp_path)
        self.cleanup(path = self.save_path)

    def download(self, url: str, path: str, chunk_size: int) -> None:
        response = requests.get(url = url, stream = True)
        try:
            content_length = int(response.headers['Content-Length'])
        except KeyError:
            content_length = None
        with open(path, 'wb') as zip:
            progress_bar = tqdm(total = content_length)
            for chunk in response.iter_content(chunk_size = chunk_size):
                zip.write(chunk)
                progress_bar.update(chunk_size)

    def unzip(self, origin_path: str, destination_path: str = None) -> None:
        if destination_path is None:
            destination_path = os.path.dirname(origin_path)
        with zipfile.ZipFile(origin_path, 'r') as zip:
            zip.extractall(destination_path)

    def cleanup(self, path: str) -> None:
        pass