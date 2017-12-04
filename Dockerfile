# TODO(adamb) This is awful. Instead cache all files and install that way.
FROM ubuntu:16.04

# Set the ENTRYPOINT to use bash
# (this is also where you’d set SHELL,
# if your version of docker supports this)
# ENTRYPOINT [ "/bin/bash", "-c" ]


###############################
## A little Docker magic here

# Force bash always
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
# Default miniconda installation
ENV CONDA_ENV_PATH /opt/miniconda
# This is how you will activate this conda environment
# "source $CONDA_ENV_PATH/bin/activate $MY_CONDA_PY3ENV"

################
# Conda supports delegating to pip to install dependencies
# that aren’t available in anaconda or need to be compiled
# for other reasons.
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
ADD environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN ["conda", "env", "create"]

# Should really be in environment.yml, but don't want to reupload it right now.
RUN source /opt/miniconda/bin/activate someenv && pip install cloudpickle

ADD docker/conda-entrypoint.sh $CONDA_ENV_PATH/bin/entrypoint.sh
ADD src /code/src

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /code/src
ENV PATH /code/src/tfi:$PATH

# ENV PYTHONPATH /code/src:/code
# ADD zoo /code/zoo
# This should be done as a separate command and uploaded. Do it here for now.
# RUN mkdir /zoo && \
#     env TORCH_MODEL_ZOO=/code/zoo/torchvision \
#         entrypoint.sh \
#         --export /zoo/resnet50.tfi \
#         zoo.torchvision.resnet.Resnet50 && \
#     rm -R /code/zoo

WORKDIR /code
# RUN useradd -m srv
# USER srv

ENTRYPOINT ["entrypoint.sh"]
CMD ["--bind=0.0.0.0:8888", "--serve"]
# CMD ["--bind=0.0.0.0:5000", "--serve", "@/zoo/resnet50.tfi"]
