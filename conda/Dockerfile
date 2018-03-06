# TODO(adamb) This is awful. Instead cache all files and install that way.
FROM ubuntu:16.04

###############################
## A little Docker magic here

# Force bash always
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
# Default miniconda installation
ENV CONDA_ENV_PATH /opt/miniconda

################
RUN apt-get update && apt-get install -y \
 build-essential bzip2 wget libglib2.0-0 libgl1-mesa-glx libxext6 libsm6 libxrender1 \
&& rm -rf /var/lib/apt/lists/*


###############################
# (mini)CONDA package manager

# Download and create
RUN wget --quiet \
    https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh && \
    bash Miniconda-latest-Linux-x86_64.sh -b -p $CONDA_ENV_PATH && \
    rm Miniconda-latest-Linux-x86_64.sh && \
    chmod -R a+rx $CONDA_ENV_PATH
ENV PATH $CONDA_ENV_PATH/bin:$PATH

#######################
# Install modules and dependencies
ARG CONDA_ENV_NAME

ADD conda/$CONDA_ENV_NAME.yml /tmp/environment.yml
WORKDIR /tmp
RUN ["conda", "env", "create"]

ADD conda/entrypoint.sh $CONDA_ENV_PATH/bin/entrypoint.sh
ADD src /code/src

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /code/src
ENV PATH /code/src/tfi:$PATH

WORKDIR /code
# RUN useradd -m srv
# USER srv

ENV CONDA_ENV_NAME ${CONDA_ENV_NAME}
ENTRYPOINT ["entrypoint.sh"]
CMD ["--bind=0.0.0.0:8888", "--serve", "--internal-config=${CONDA_ENV_NAME}"]
