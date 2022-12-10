# Docker file for the car acceptability prediction project
# Alex Taciuk, Dec, 2022

# use continuumio/miniconda3 as the base image 
FROM continuumio/miniconda3@sha256:977263e8d1e476972fddab1c75fe050dd3cd17626390e874448bd92721fd659b

# put anaconda python in path
ENV PATH="/opt/conda/bin:${PATH}"

# install python packages via pip
RUN pip install docopt-ng==0.8.1 \
    && pip install vl-convert-python==0.5.0 \
    && pip install docopt==0.6.2

# update available softwares
RUN apt-get update
RUN apt-get install libfontconfig1-dev -y

# install R and R packages 
RUN conda install -c conda-forge r r-essentials
RUN conda install -c conda-forge r-kableextra -y
RUN apt-get install pandoc -y
RUN Rscript -e "install.packages('kableExtra',repos = 'http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('xfun',repos = 'http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('vctrs',repos = 'http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('pandoc',repos = 'http://cran.us.r-project.org')"
RUN apt-get install libxt6

# install python packages via conda
RUN conda install -c conda-forge matplotlib -y 
RUN conda install -c conda-forge eli5 -y 
RUN conda install -c conda-forge shap -y 
RUN conda install -c conda-forge imbalanced-learn -y 
RUN conda install python-graphviz -y 
RUN conda install requests[version='>=2.24.0'] -y 
RUN conda install scikit-learn -y 
RUN conda install selenium[version='<4.3.0'] -y 
RUN conda install lightgbm -y 
RUN conda install pip -y 
RUN conda install jinja2 -y 
RUN conda install ipykernel -y 
RUN conda install jsonschema=4.16 -y 
RUN conda install -c conda-forge altair_saver -y 
RUN conda install pandas[version='<1.5'] -y 
