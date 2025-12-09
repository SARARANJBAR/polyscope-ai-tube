FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# set web proxy
ENV http_proxy http://1mcwebproxy01.mdanderson.edu:3128
ENV https_proxy http://1mcwebproxy01.mdanderson.edu:3128
ENV HTTP_PROXY http://1mcwebproxy01.mdanderson.edu:3128
ENV HTTPS_PROXY http://1mcwebproxy01.mdanderson.edu:3128
# set environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV HDF5_USE_FILE_LOCKING FALSE
ENV NUMBA_CACHE_DIR /tmp
# install libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libgl1-mesa-glx libglib2.0-0 libgeos-dev libvips-tools \
  sudo curl wget htop vim ca-certificates python3-openslide python3-pip python3-dev \
  && rm -rf /var/lib/apt/lists/*
# install python packages
RUN pip install gpustat==0.6.0 setuptools==61.2.0 pytz==2023.3 joblib==1.2.0 future==0.18.2 docopt==0.6.2
RUN pip install tqdm==4.66.1 mako==1.2.4 einops==0.6.1 geojson==3.0.0 h5py==3.9.0 kaleido==0.2.1 natsort==8.4.0
RUN pip install ipython==8.10.0 jupyterlab==3.6.1 notebook==6.4.11 traitlets==5.9.0 chardet==5.0.0 nbconvert==7.8.0 pyjwt==2.6.0
RUN pip install pandas==2.2.2 pandarallel==1.6.5 matplotlib==3.9.1 seaborn==0.13.0 pycm==3.5 deepdish==0.3.7 openpyxl==3.1.2 XlsxWriter==3.2.0
RUN pip install numpy==1.24.4 scikit-learn==1.3.2 xgboost==2.0.3 statsmodels==0.13.5 lifelines==0.27.8 rasterio==1.3.5.post1 ujson==5.8.0
RUN pip install openslide-python==1.3.1 Pillow==10.0.0 opencv-python==4.8.0.74 scikit-image==0.24.0 imgaug==0.4.0 mahotas==1.4.12 Shapely==1.8.5.post1
RUN pip install imagecodecs==2024.6.1 tifffile==2024.5.22 pydantic==1.10.4 imageio==2.31.1 csbdeep==0.7.4 stardist==0.8.5 albumentations==1.3.0 histolab==0.6.0
RUN pip install torch==2.6.0 torchvision torchaudio==2.6.0 torchinfo==1.8.0 torchmetrics==1.4.0
RUN pip install cucim-cu12==24.2.0 cupy-cuda12x==12.0.0
RUN pip install tensorflow==2.13.0
RUN pip install tensorflow-hub==0.16.1 tensorflow-datasets==4.9.4 tensorboard==2.17.0
RUN pip install scanpy==1.10.2 geopandas==1.0.1 anndata==0.10.8
RUN pip install pyg-nightly==2.4.0.dev20231209 networkx==3.4.2 pysal==23.7 spatialentropy==0.0.4
RUN pip install transformers==4.36.2 pathologyfoundation==0.1.14 ultralytics==8.0.230 accelerate==0.33
RUN pip install timm==0.9.8
RUN pip install igraph==0.11.8 leidenalg==0.10.2
RUN pip install tdigest==0.5.2 datasets==3.1.0 hyperopt==0.2 loompy==3.0.6 optuna==3.6 optuna-integration==3.6 packaging==23.0
RUN pip install peft==0.11.1 pyarrow==14.0.1 ray==2.8.1
RUN pip install hdbscan==0.8.40
RUN pip install pyvips==2.2.3
# New inclusions and updates for SAM2
RUN pip install hydra-core==1.3.2 iopath==0.1.10 pillow==10.2.0
#RUN pip install jupyter==1.0.0 eva-decord==0.6.1
#RUN pip install Flask==3.0.3 Flask-Cors==5.0.0 av==13.0.0 dataclasses-json==0.6.7 gunicorn==23.0.0
#RUN pip install imagesize==1.4.1 pycocotools==2.0.8 strawberry-graphql==0.243.0
#RUN pip install black==24.2.0 usort==1.0.2 ufmt==2.0.0b2 fvcore==0.1.5.post20221221
#RUN pip install tensordict==0.6.0 submitit==1.5.1
# configure folder permission
WORKDIR /.dgl
RUN chmod -R 777 /.dgl
WORKDIR /.local
RUN chmod -R 777 /.local
WORKDIR /.cache
RUN chmod -R 777 /.cache
# set working directory 
WORKDIR /App
CMD ["/bin/bash"]
# CMD["/usr/bin/python3", "python_script.py"]