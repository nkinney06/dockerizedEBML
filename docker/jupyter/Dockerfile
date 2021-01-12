#FROM jupyter/scipy-notebook
FROM jupyter/datascience-notebook
USER root

RUN apt-get update && apt-get dist-upgrade -y && apt clean all
RUN apt-get update && apt-get install -y p7zip-full && apt-get clean all

RUN conda install --yes --name root psycopg2
RUN conda install --yes assimulo
RUN ["bash", "-c", "source activate root && pip install modelbase"] 

RUN R -e "install.packages('VennDiagram',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('ggplot2',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('ggpubr',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('plyr',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('dplyr',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('vcd',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('ggbeeswarm',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('vcdExtra',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('BiocManager',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "BiocManager::install('DESeq2')"
RUN R -e "BiocManager::install('org.Hs.eg.db')"
RUN R -e "BiocManager::install('EnhancedVolcano')"

USER jovyan 
