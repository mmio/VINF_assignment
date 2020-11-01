import os
import urllib.request
from files import urls, destination

def download_file(url, filename):
    name, _ = urllib.request.urlretrieve(url, filename)
    return open(name, 'rb')

def download_and_save(url):
    file = url.split('/')[-1]
    filepath = f'{destination}/{file}'

    if file in os.listdir(destination):
        return open(filepath, 'rb')
    else:
        return download_file(url, filepath)