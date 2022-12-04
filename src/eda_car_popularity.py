# author: Lisa Sequeira, Alex Taciuk, Peng Zhang, Gaoxiang Wang
# date: 2022-11-20
'''This script takes the file path of training data, then exports all EDA plots and
then saves them to the result folder.

Usage: src/eda_car_popularity.py --training_data_path=<training_data_path> --folder_result_path=<folder_result_path>

Options: 
--training_data_path=<training_data_path> The path the the csv file containing training data
--folder_result_path=<folder_result_path> The path to store the eda plots
'''
# To execute script below run the following in your terminal
# python eda_car_popularity.py --training_data_path=../data/processed/training.csv --folder_result_path=../result
from docopt import docopt 
import altair as alt
import pandas as pd
import os
import vl_convert as vlc

# reference to DSCI 531
# Save a vega-lite spec and a PNG blob for each plot in the notebook
alt.renderers.enable('mimetype')

opt = docopt(__doc__)

def main(training_data_path, folder_result_path):
    try:
        # read data and add column names 
        train_df = pd.read_csv(training_data_path)
    except Exception as req:
        print("The training data file path does not exist, check again")
        print(req)

    # Plot 1: Generate dist plot showing relation between count of each feature
    # category against target
    corr_plot = alt.Chart(train_df).mark_square().encode(
        x=alt.X(alt.repeat(), sort='-color'),
        y=alt.Y('class', sort='color'),
        color='count()',
        size='count()').properties(
        width=100,
        height=100
    ).repeat(
    ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],columns = 2
    )
    #Export plot 1:
    try:
        corr_plot.save(folder_result_path+'/eda_plot1_corr.png')
    except:
        os.makedirs(os.path.dirname(folder_result_path), exist_ok=True)
        corr_plot.save(folder_result_path+'/eda_plot1_corr.png') 

    #Plot 2: Histograms showing distribution of each feature
    features = train_df.columns.tolist()
    dist_plot = alt.Chart(train_df).mark_bar().encode(
        alt.X(alt.repeat()
        ),
        y='count()'
        ).repeat(features, columns=3)
    
    #Export plot 2:
    try:
        dist_plot.save(folder_result_path+'/eda_plot2_dist.png')
    except:
        os.makedirs(os.path.dirname(folder_result_path), exist_ok=True)
        dist_plot.save(folder_result_path+'/eda_plot2_dist.png')

#Function to output the plot in the folder_results_path
def plot_save(plot, filename:str, folder_path):
    png_data = vlc.vegalite_to_png(plot.to_dict())
    
    #Writing the plot to the folder path
    #Example: with open(folder_result_path+"/chart.png", "wb")
    with open(folder_path+filename, "wb") as f:
        f.write(png_data)
    
    #Because of an altair issue, we cannot use the code line below to export graphs
        #try:
            #distri_plot.save(folder_result_path+'/EDA_distribution_plot.png')
        #except:
            #os.makedirs(os.path.dirname(folder_result_path), exist_ok=True)
            #distri_plot.save(folder_result_path+'/EDA_distribution_plot.png')

if __name__ == "__main__":
    main(opt["--training_data_path"], opt["--folder_result_path"])