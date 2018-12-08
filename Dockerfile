ARG BASE_CONTAINER=jupyter/minimal-notebook

FROM $BASE_CONTAINER

LABEL maintainer="Alang Hsu <alang.hsu@gmail.com"

# Tensorflow must be first installation.
RUN conda install --quiet --yes \
    'tensorflow=1.11*' \
    'keras=2.2*' 

# Install python 3 packages
RUN conda install --quiet --yes \
    'pandas=0.23*' \
    'matplotlib=2.2*' \
    'seaborn=0.9*' \
    'scikit-learn=0.20*' \
    'scikit-image=0.14*' \
    'sympy=1.1*' 

# Install opencv
RUN pip install opencv-python imutils


# Cleaning up the unused files
RUN conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

