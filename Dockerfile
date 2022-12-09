FROM continuumio/miniconda3@sha256:977263e8d1e476972fddab1c75fe050dd3cd17626390e874448bd92721fd659b

RUN pip install docopt-ng==0.8.1 \
    && pip install vl-convert-python==0.5.0

RUN conda install python-graphviz -y \
    && conda install requests[version='>=2.24.0'] -y \
    && conda install scikit-learn -y \
    && conda install selenium[version='<4.3.0'] -y \
    && conda install lightgbm -y \
    && conda install pip -y \
    && conda install jinja2 -y \
    && conda install ipykernel -y \
    && conda install jsonschema=4.16 -y \
    && conda install -c conda-forge altair_saver -y \
    && conda install pandas[version='<1.5'] -y \
    && conda install matplotlib[version='>=3.2.2'] -y \
    && conda install graphviz -y \
    && conda install -c anaconda docopt -y \
    && conda install -c conda-forge eli5 -y \
    && conda install -c conda-forge shap -y \
    && conda install -c conda-forge imbalanced-learn -y 

RUN apt-get update
RUN apt-get install libfontconfig1-dev -y

# install R and R packages 
RUN conda install -c conda-forge r r-essentials
RUN Rscript -e "install.packages('kableExtra',repos = 'http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('pandoc',repos = 'http://cran.us.r-project.org')"
