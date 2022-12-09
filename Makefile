# Car-Acceptability-Prediction
# Author: DSCI 522 Group 19
# Date: 2022-12-03

all: doc/car_popularity_prediction_report.html doc/car_popularity_prediction_report.md

# download raw data
data/raw/raw_data.csv: src/download_data.py
	python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data --local_file_path=data/raw/raw_data.csv

# preprocessing of the data and data splitting into train and test
data/processed/training.csv data/processed/test.csv: src/data_processing.py data/raw/raw_data.csv
	python src/data_processing.py --raw_data_path=data/raw/raw_data.csv --clean_data_folder_path=data/processed

# exploratory data analysis - visualize relationship between features and between target to features
result/eda_plot1_corr.png result/eda_plot2_dist.png: src/eda_car_popularity.py data/processed/training.csv
	python src/eda_car_popularity.py --training_data_path=data/processed/training.csv --folder_result_path=result

# Tuning, training and testing classifier model
result/car_classifier_tuning_analysis.csv result/car_hyperparameter_tuning_analysis.csv result/final_model.joblib result/final_model_score.txt: src/car_classifier_analysis.py data/processed/training.csv data/processed/test.csv
	python src/car_classifier_analysis.py --train_test_folder_path=data/processed --result_folder_path=result

# render report
doc/car_popularity_prediction_report.html doc/car_popularity_prediction_report.md: doc/car_popularity_prediction_report.Rmd doc/references.bib result/eda_plot2_dist.png result/eda_plot1_corr.png result/car_classifier_tuning_analysis.csv result/car_hyperparameter_tuning_analysis.csv 
	Rscript -e "rmarkdown::render('doc/car_popularity_prediction_report.Rmd', output_format = 'github_document')"

clean: 
	rm -rf data
	rm -rf result
	rm -rf doc/car_popularity_prediction_report.md doc/car_popularity_prediction_report.html