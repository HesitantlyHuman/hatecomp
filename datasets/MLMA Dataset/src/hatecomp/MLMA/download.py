import requests
import zipfile
import os

DOWNLOAD_URL = 'https://github.com/HKUST-KnowComp/MLMA_hate_speech/raw/master/hate_speech_mlma.zip'
DOWNLOAD_DIR = './MLMA Dataset/data'
FILENAME = 'hate_speech_mlma.zip'

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

def download_MLMA(
    save_path = None,
    unzip = True,
    keep_zip = False,
    chunk_size = 128
):
    if save_path is None:
        if not os.path.exists(DOWNLOAD_DIR):
            os.mkdir(DOWNLOAD_DIR)
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
    download_MLMA()