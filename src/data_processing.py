# author: Lisa Sequeira, Alex Taciuk, Peng Zhang, Gaoxiang Wang
# date: 2022-11-20
'''This script takes the file path of raw data, then add column names and split the data into training/test data with seed, 
write the two data to the processed/clean data folder path.

Usage: src/data_processing.py --raw_data_path=<raw_data_path> --clean_data_folder_path=<clean_data_folder_path>

Options: 
--raw_data_path=<raw_data_path> The path the the csv file contianing raw data
--clean_data_folder_path=<clean_data_folder_path> The Folder path to store the training and test csv files
'''

from docopt import docopt 
import os 
import pandas as pd 
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(raw_data_path, clean_data_folder_path):
    try:
        # read data and add column names 
        data = pd.read_csv(raw_data_path, names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'] , header = 0)
    except Exception as req:
        print("The raw data file path does not exist, check again")
        print(req)
    # split training/test data with reproducibility
    train_df, test_df = train_test_split(data, test_size=0.10, random_state=522)
    write_to_csv(train_df, clean_data_folder_path+'/training.csv')
    write_to_csv(test_df, clean_data_folder_path+'/test.csv')

# function for writing csv file to given folder path
def write_to_csv(df, file_path):
    try:
        df.to_csv(file_path, index = False)
    except:
        os.makedirs(os.path.dirname(file_path))
        df.to_csv(file_path, index = False)



if __name__ == "__main__":
    main(opt["--raw_data_path"], opt["--clean_data_folder_path"])

