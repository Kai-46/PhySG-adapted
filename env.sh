export ENV_NAME=physg
conda create -n ${ENV_NAME} python=3.10 -y
conda activate ${ENV_NAME}

if [ ! -d /usr/local/cuda-11.5 ]; then
    CUR_DIR=$(pwd)
    TEMP_DIR=/mnt/localssd/temp
    mkdir -p ${TEMP_DIR}

    CUDA_HOME_EXPORT="export CUDA_HOME=/usr/local/cuda"
    PATH_EXPORT="export PATH=/usr/local/cuda/bin:\$PATH"
    LD_LIBRARY_PATH_EXPORT="export LD_LIBRARY_PATH=usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    SHELL_PROFILE=~/.bashrc
    if ! grep -q ${CUDA_HOME_EXPORT} ${SHELL_PROFILE}; then
        echo -e "\n\n" >> ${SHELL_PROFILE}
        echo ${CUDA_HOME_EXPORT} >> ${SHELL_PROFILE}
        echo ${PATH_EXPORT} >> ${SHELL_PROFILE}
        echo ${LD_LIBRARY_PATH_EXPORT} >> ${SHELL_PROFILE}
        source ${SHELL_PROFILE} && conda activate ${ENV_NAME}  # just in case source ~/.bashrc activates base env
    fi

    cd ${TEMP_DIR} && \
        wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda_11.5.0_495.29.05_linux.run && \
        sudo sh cuda_11.5.0_495.29.05_linux.run --silent --override --toolkit --toolkitpath=/usr/local/cuda-11.5 --no-opengl-libs && \
        sudo chmod -R 777 /usr/local/cuda-11.5
    [ -e /usr/local/cuda ] && sudo rm /usr/local/cuda
    sudo ln -s /usr/local/cuda-11.5 /usr/local/cuda
    nvcc --version
    cd ${CUR_DIR}
fi


pip install ninja \
            trimesh \
            opencv-python \
            imageio \
            tensorboardX \
            torch \
            torch-ema \
            torchmetrics \
            numpy \
            pandas \
            tqdm \
            matplotlib \
            PyMCubes \
            rich \
            pysdf \
            scipy \
            lpips \
            isort \
            black \
            gdown \
            accelerate \
            "pyglet<2" \
            plotly \
            kaleido \
            libigl \
            GPUtil \
            pyhocon==0.3.55 \
            scikit-image \
            cvxpy
