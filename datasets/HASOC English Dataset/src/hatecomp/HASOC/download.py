import requests
import zipfile
import os

DOWNLOAD_URL = 'https://hasocfire.github.io/hasoc/2019/files/english_dataset.zip'
DOWNLOAD_DIR = './HASOC English Dataset/data'
FILENAME = 'english_dataset.zip'

def _download_zip(
    save_path = None,
    chunk_size = 128
):
    response = requests.get(DOWNLOAD_URL, stream = True)
    with open(save_path, 'wb') as zip:
        for chunk in response.iter_content(chunk_size = chunk_size):
            zip.write(chunk)

def _unzip(
    path,
    location = None
):
    if location is None:
        location = os.path.dirname(path)
    with zipfile.ZipFile(path, 'r') as zip:
        zip.extractall(location)

#[TODO]: HASOC has a lot of unecessary files we should delete
def _cleanup_HASOC(
    path
):
    pass

def download_HASOC(
    save_path = None,
    unzip = True,
    keep_zip = False,
    chunk_size = 128
):
    if save_path is None:
        save_path = os.path.join(DOWNLOAD_DIR, FILENAME)

    _download_zip(
        save_path = save_path,
        chunk_size = chunk_size
    )
    if unzip:
        _unzip(save_path)
    if not keep_zip:
        os.remove(save_path)

if __name__ == '__main__':
    download_HASOC()