import os
import gdown
import zipfile


DATASET_URL = {
    'crack': 'https://drive.google.com/file/d/1IBJ7rtsU17p0yBZl2jpuGd6isDr1Rnn-/view?usp=share_link',
}


def download(dataset_name):
    if not dataset_name in DATASET_URL:
        raise Exception('dataset not found.')
    
    if not os.path.exists('datasets'):
        os.mkdir('datasets')

    url = DATASET_URL[dataset_name]
    destination = 'datasets/'

    # download dataset (.zip) from google drive
    path_to_zip = gdown.download(url, output=destination, quiet=False, fuzzy=True)

    # unzip
    with zipfile.ZipFile(path_to_zip, 'r') as zip:
        zip.extractall(destination)

    # delete zip file
    os.remove(path_to_zip)
    