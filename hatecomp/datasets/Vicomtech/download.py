import os
import csv
import shutil

from hatecomp.base.download import _ZipDownloader
from hatecomp._path import install_path


class VicomtechDownloader(_ZipDownloader):
    DOWNLOAD_URL = (
        "https://github.com/Vicomtech/hate-speech-dataset/archive/refs/heads/master.zip"
    )

    VALID_LABELS = ["noHate", "hate"]

    def __init__(self, save_path) -> None:
        url = self.DOWNLOAD_URL
        super().__init__(url, save_path)

    def cleanup(self, path: str) -> None:
        annotation_path = os.path.join(
            path, "hate-speech-dataset-master/annotations_metadata.csv"
        )
        with open(annotation_path, "r") as annotation_csv_file:
            annotation_csv = list(csv.reader(annotation_csv_file))
        vicomtech_data = []
        ids = [item[0] for item in annotation_csv[1:]]
        content = self.get_content(path, ids)
        for item, content_item in zip(annotation_csv[1:], content):
            item = ["Vicomtech-" + item[0], content_item, item[4]]
            vicomtech_data.append(item)
        save_location = os.path.join(path, "vicomtech.csv")
        with open(save_location, "w") as csv_file:
            writer = csv.writer(csv_file)
            keys = ["id", "data", "hate"]
            writer.writerow(keys)
            for item in vicomtech_data:
                if item[2] in self.VALID_LABELS:
                    writer.writerow(item)
        shutil.rmtree(os.path.join(path, "hate-speech-dataset-master"))

    def get_content(self, path, ids):
        content = []
        for id in ids:
            file_path = f"hate-speech-dataset-master/all_files/{id}.txt"
            file_location = os.path.join(path, file_path)
            with open(file_location, "r") as content_file:
                content.append(content_file.read())
        return content
