# Usage

Note: Make sure have the necessary dependencies installed for your conda environment.

1. Using `docopt` in Python to download data file from external URL 

To proceed, run the following command at the command line and set root directory as the current working directory.

`python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data --local_file_path="data/raw/raw_data.csv"`

2. Using `Altair` in Python to render the literate Exploratory Data Analysis.

To proceed, open and run "src/car_popularity_analysis.ipynb"


# Dependencies

* Python 3.10.6 and Python Packages:
    * Sub docopt-ng                 0.8.1 
    * Sub requests                  2.28.1 
    * Sub pandas                    1.5.1
    * Sub altair                    4.2.0 
    * Sub altair_data_server        0.4.1
    * Sub sklearn                   1.1.3
