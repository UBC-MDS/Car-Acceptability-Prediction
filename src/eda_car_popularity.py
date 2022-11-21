# date: 2022-11-20
'''This script takes the file path of traning data, then plot the side by side counting combination 
for all categorical groups by target, then save the plot to result folder.

Usage: src/eda_car_popularity.py --training_data_path=<training_data_path> --plot_result_path=<plot_result_path>

Options: 
--training_data_path=<training_data_path> The path the the csv file contianing training data
--plot_result_path=<plot_result_path> The path to store the eda plots
'''

from docopt import docopt 
import altair as alt
import pandas as pd
import os
# reference to DSCI 531
# Save a vega-lite spec and a PNG blob for each plot in the notebook
alt.renderers.enable('mimetype')
# Handle large data sets without embedding them in the notebook
alt.data_transformers.enable('data_server')
opt = docopt(__doc__)

def main(training_data_path, plot_result_path):
    try:
        # read data and add column names 
        train_df = pd.read_csv(training_data_path)
    except Exception as req:
        print("The training data file path does not exist, check again")
        print(req)
    # generate side by side plots with 2 columns
    distri_plot = alt.Chart(train_df).mark_square().encode(
        x=alt.X(alt.repeat(), sort='-color'),
        y=alt.Y('class', sort='color'),
        color='count()',
        size='count()').properties(
        width=100,
        height=100
    ).repeat(
    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],columns = 2
    )
    try:
        distri_plot.save(plot_result_path)
    except:
        os.makedirs(os.path.dirname(plot_result_path))
        distri_plot.save(plot_result_path)
    

if __name__ == "__main__":
    main(opt["--training_data_path"], opt["--training_data_path"])