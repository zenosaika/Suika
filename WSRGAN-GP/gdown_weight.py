import os
import gdown
import zipfile


WEIGHT_URL = {
    'wsrgan-gp': 'https://drive.google.com/file/d/18P4PjhaltGqcdYePjDdm9thtuJ9pktHX/view?usp=share_link',
}


def download(weight_name, weights_dir):
    if not weight_name in WEIGHT_URL:
        raise Exception('weight not found.')
    
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    url = WEIGHT_URL[weight_name]
    destination = weights_dir + '/'

    # download weight (.zip) from google drive
    path_to_zip = gdown.download(url, output=destination, quiet=False, fuzzy=True)

    # unzip
    with zipfile.ZipFile(path_to_zip, 'r') as zip:
        zip.extractall(destination)

    # delete zip file
    os.remove(path_to_zip)
    