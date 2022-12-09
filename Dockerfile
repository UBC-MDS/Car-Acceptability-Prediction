FROM continuumio/miniconda3@sha256:977263e8d1e476972fddab1c75fe050dd3cd17626390e874448bd92721fd659b

# activate the environment
COPY ./env522car.yaml /opt/
RUN conda env create -f /opt/env522car.yaml
RUN echo "source activate 522env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH    

RUN apt-get update
RUN apt-get install libfontconfig1-dev -y

# install R and R packages 
RUN conda install -c conda-forge r r-essentials
RUN Rscript -e "install.packages('kableExtra',repos = 'http://cran.us.r-project.org')"
RUN Rscript -e "install.packages('pandoc',repos = 'http://cran.us.r-project.org')"
