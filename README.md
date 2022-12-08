# Car Popularity Prediction

**Authors:** Lisa Sequeira, Alex Taciuk, Peng Zhang, Gaoxiang Wang

**Our research question: Given attributes of car models from the 90s, can we predict how popular each car model will be?**

In order to pay for MDS tuition, our team is starting a side-business buying and selling used cars.  Used cars have seen recent high demand and varied prices so we think this is a fertile ground for a price arbitrage business. In order to run a successful business buying and selling used cars, we need to determine the most popular used car models. 

The cars we are aiming to buy and sell will be older (ideally from the 90s), so that we can afford to buy and sell them. We will use a dataset of car popularity by model primarily collected in June, 1997 by Marko Bohanec to evaluate HINT (Hierarchy INduction Tool), and is presented in B. Zupan, M. Bohanec, I. Bratko, J. Demsar: Machine learning by function decomposition. ICML-97, Nashville, TN. 1997. The data can be accessed from the [UC Irvine Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/19/car+evaluation) and follows a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/legalcode). Each row of this dataset is a car model, with popularity. 

Using the above dataset, we will create a model to predict how popular cars were among drivers, that is generalizable to older car models beyond this dataset.  Specifically, we want to predict the correct category (or classification) of car popularity/acceptability (class) based on six commonly available  car attributes (features):
* Price
* Price of maintenance
* Number of doors
* Capacity of the car (number of passengers we can carry)
* Size of trunk or luggage space
* Estimated safety of the car  


The first thing we will do is split our data into test and training sets (with a 90% to 10% split). Next we will perform some exploratory data analysis on the train dataset, by looking to see if any features contain any missing or null values, their respective data types, the split in our target categories (to watch out for any class imbalances). 

We will then perform exploratory data analysis (EDA). We will review the distribution of our target class (to see if we have a class imbalance) and compare each of the features to the target class to see if there are any relationships or strong correlations between them. We will also evaluate how “dirty” our data is so that we know what types of transformations we will need to perform. Using our initial EDA will make an assessment on the models we want to use. 

Next we will begin pre-processing and data cleaning our training data by creating a column transformer to add into our pipeline. Given our features contain a lot of categorical variables and the final predicted target is categorical, it might make sense to use a decision tree classifier model or K-NN classifier. A potential baseline model we could use to start with could be a DummyClassifier. For each of our selected models we will conduct hyperparameter optimization and compare classification metrics (confusion matrix, precision, recall and F1 scores) to select the optimal model for downstream testing. This will be presented as a table and graphs in the analysis. 

We will share the results of our analysis via the following: After selecting our optimal model, we will use the pipe line created above to re-run these on our test dataset and then analyze the scores from the prediction. We will once again review and present the classification metrics using tables and graphs and and decide if our model is good enough to use on deployment data. 



# Usage
Note: Make sure have the necessary dependencies installed for your conda environment.

Fork the repo and clone the forked repo to your local machine by runng `git clone {repo URL}`. Then run `conda env create -f env522car.yaml` and `conda activate 522env`.
## using `make`

To reproduce the analysis, run the command at the root directory.

`make all`

To reset and redo the analysis from beginning, run the command at the root directory.

`make clean`
## without using `make`

1. Using `docopt` in Python to download data file from external URL 

To proceed, run the following command at the command line and set root directory as the current working directory.

`python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data --local_file_path=data/raw/raw_data.csv`

2. Using `Altair` in Python to render the literate Exploratory Data Analysis.

To proceed, open and run `src/car_popularity_analysis.ipynb`

3. For processing raw data, run the following command at the root directory.

`python src/data_processing.py --raw_data_path=data/raw/raw_data.csv --clean_data_folder_path=data/processed`

4. For eda plotting of training data, run the following command at the root directory.

`python src/eda_car_popularity.py --training_data_path=data/processed/training.csv --folder_result_path=result`

5. For machine learning analysis for traning data, run the following command at the root directory.

`python src/car_classifier_analysis.py --train_test_folder_path=data/processed --result_folder_path=result`

6. To render final report, run the following command at the project directory.

`Rscript -e "rmarkdown::render('doc/car_popularity_prediction_report.Rmd')"`

# Dependencies

The conda environment, called `env522car.yaml`, can be found in the root directory.

* Python 3.10.6 and Python Packages:
    *  docopt-ng                 0.8.1 
    *  requests                  2.28.1 
    *  pandas                    1.5.1
    *  altair                    4.2.0 
    *  altair_data_server        0.4.1
    *  joblib                    1.2.0
    *  matplotlib                >=3.2.2
    *  jsonschema                4.16
    *  selenium                  <4.3.0
    *  docopt-ng                 0.8.1
    *  vl-convert-python         0.5.0
    *  eli5                      0.13.0
    *  shap                      0.41.0
    *  imbalanced-learn          0.9.1
    *  lightgbm                  3.3.3
    *  pip                       22.3.1
    *  jinja2                    3.1.2
    *  ipykernel                 6.18.0
    *  altair_saver              0.5.0
    *  python-graphviz           0.20.1
    *  graphviz                  6.0.1
    *  scikit-learn              1.1.3          

# License

The Car Popularity Predictor materials here are licensed under [Creative Commons Attribution 4.0 Canada License (CC BY 4.0 CA).](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode) 
