# author: Gaoxiang Wang
# date: 2022-11-17
'''This script takes a URL to download data files from the internet and writes the files to a local filepath.

Usage: src/download_data.py --url=<url> --local_file_path=<local_file_path>

Options: 
--url=<url> URL of the data from internet (has to be csv readable)
--local_file_path=<local_file_path> Location (file path and file name) to store the csv file from URL
'''

from docopt import docopt
import requests
import os
import pandas as pd

opt = docopt(__doc__)

def main(url, local_file_path):
    try: 
        request = requests.get(url)
        request.status_code == 200
    except Exception as req:
        print("URL is not working, try different one")
        print(req)

    data = pd.read_csv(url,delimiter=",", header = None)
        
    try:
        data.to_csv(local_file_path, index = False)
    except:
        os.makedirs(os.path.dirname(local_file_path))
        data.to_csv(local_file_path, index = False)

if __name__ == "__main__":
    main(opt["--url"], opt["--local_file_path"])
