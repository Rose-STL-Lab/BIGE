################## How to push to Nautilus Gitlab #####################
# docker login gitlab-registry.nrp-nautilus.io
# docker tag digital-coach-anwesh:latest gitlab-registry.nrp-nautilus.io/shmaheshwari/digital-coach-anwesh:latest
# docker build -t gitlab-registry.nrp-nautilus.io/shmaheshwari/digital-coach-anwesh:latest .
# docker push gitlab-registry.nrp-nautilus.io/shmaheshwari/digital-coach-anwesh:latest

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Install ALL base dependencies and CMake in one layer
RUN apt-get update && apt-get install -y \
    wget \
    git \
    htop \
    curl \
    build-essential \
    make \
    cmake \
    sudo \
    libtinyxml2-dev \
    libgmp3-dev \
    libgmp-dev \
    yasm \
    m4 \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libboost-all-dev \
    xvfb \
    xauth \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libxrender1 \
    libxrandr2 \
    libxrandr-dev \
    libxinerama-dev \
    libxss1 \
    libxcursor1 \
    libxcursor-dev \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxi-dev \
    libxtst6 \
    lsof \
    autoconf \
    automake \
    libtool \
    freeglut3-dev \
    && echo "🔧 Installing CMake 3.31.5..." \
    && cd /tmp \
    && wget https://github.com/Kitware/CMake/releases/download/v3.31.5/cmake-3.31.5-linux-x86_64.sh \
    && chmod +x cmake-3.31.5-linux-x86_64.sh \
    && ./cmake-3.31.5-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.31.5-linux-x86_64.sh \
    && cmake --version \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Set environment variables
ENV PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Install Miniconda and initialize
RUN cd /tmp \
    && echo "🔧 Installing Miniconda..." \
    && wget -q https://repo.continuum.io/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh \
    && chmod +x Miniconda3-py38_23.1.0-1-Linux-x86_64.sh \
    && ./Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b -f -p /usr/local \
    && rm Miniconda3-py38_23.1.0-1-Linux-x86_64.sh \
    && conda init bash \
    && conda clean -afy \
    && rm -rf /tmp/*

ENV PATH=/usr/local/bin:$PATH

# Install core math and geometry libraries (Eigen, CCD, ASSIMP)
RUN cd /tmp \
    && echo "🔧 Installing core math libraries (Eigen, CCD, ASSIMP)..." \
    && curl https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz | tar -zx \
    && cd eigen-3.3.7 \
    && mkdir build && cd build \
    && cmake -DENABLE_DOUBLE_PRECISION=ON -DBUILD_SHARED_LIBS=ON .. \
    && make -j$(nproc) && make install \
    && cd /tmp && rm -rf eigen-3.3.7 \
    && git clone --depth 1 https://github.com/danfis/libccd.git \
    && cd libccd \
    && mkdir build && cd build \
    && cmake -DENABLE_DOUBLE_PRECISION=ON -DBUILD_SHARED_LIBS=ON .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf libccd \
    && git clone --depth 1 --branch v5.0.0 https://github.com/assimp/assimp.git \
    && cd assimp \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf assimp \
    && ldconfig \
    && rm -rf /tmp/*

# Install optimization libraries (MUMPS, IPOPT)
RUN cd /tmp \
    && echo "🔧 Installing optimization libraries (MUMPS, IPOPT)..." \
    && git clone --depth 1 https://github.com/coin-or-tools/ThirdParty-Mumps.git \
    && cd ThirdParty-Mumps \
    && ./get.Mumps \
    && ./configure \
    && make -j$(nproc) \
    && make install \
    && cd /tmp && rm -rf ThirdParty-Mumps \
    && git clone --depth 1 https://github.com/coin-or/Ipopt.git \
    && cd Ipopt \
    && ./configure --with-mumps \
    && make -j$(nproc) \
    && make install \
    && ln -s /usr/local/include/coin-or /usr/local/include/coin \
    && cd /tmp && rm -rf Ipopt \
    && ldconfig \
    && rm -rf /tmp/*

# Install Python bindings and collision libraries (pybind11, FCL, octomap)
RUN cd /tmp \
    && echo "🔧 Installing Python bindings and collision libraries..." \
    && git clone --depth 1 https://github.com/pybind/pybind11.git \
    && cd pybind11 \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf pybind11 \
    && git clone --depth 1 --branch 0.3.4 https://github.com/flexible-collision-library/fcl.git \
    && cd fcl \
    && if [ -f "include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h" ]; then \
        sed -i '1696s/v0_dist/(double)v0_dist/' include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h; \
    fi \
    && mkdir build && cd build \
    && cmake .. -DFCL_WITH_OCTOMAP=OFF \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf fcl \
    && git clone --depth 1 --branch v1.9.8 https://github.com/OctoMap/octomap.git \
    && cd octomap \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf octomap \
    && git clone --depth 1 https://github.com/gonultasbu/tinyxml2 \
    && cd tinyxml2 \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf tinyxml2 \
    && ldconfig \
    && rm -rf /tmp/*

# Install graphics libraries (OpenSceneGraph)
RUN cd /tmp \
    && echo "🔧 Installing graphics libraries (OpenSceneGraph)..." \
    && git clone --depth 1 --branch OpenSceneGraph-3.6.5 https://github.com/openscenegraph/OpenSceneGraph.git \
    && cd OpenSceneGraph \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf OpenSceneGraph \
    && ldconfig \
    && rm -rf /tmp/*

# Install URDF libraries (urdfdom_headers, console_bridge, urdfdom)
RUN cd /tmp \
    && echo "🔧 Installing URDF libraries..." \
    && git clone --depth 1 https://github.com/ros/urdfdom_headers.git \
    && cd urdfdom_headers \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf urdfdom_headers \
    && git clone --depth 1 https://github.com/ros/console_bridge.git \
    && cd console_bridge \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf console_bridge \
    && git clone --depth 1 https://github.com/ros/urdfdom.git \
    && cd urdfdom \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf urdfdom \
    && ldconfig \
    && rm -rf /tmp/*

# Install performance and communication libraries (PerfUtils, Protobuf, GRPC, benchmark)
RUN cd /tmp \
    && echo "🔧 Installing performance and communication libraries..." \
    && git clone --depth 1 https://github.com/PlatformLab/PerfUtils.git \
    && cd PerfUtils \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_CXX_FLAGS="-Wno-maybe-uninitialized" \
    && make install \
    && cd /tmp && rm -rf PerfUtils \
    && PROTOBUF_VERSION="3.14.0" \
    && wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz \
    && tar -xzf protobuf-all-${PROTOBUF_VERSION}.tar.gz \
    && cd protobuf-${PROTOBUF_VERSION} \
    && ./configure \
    && make -j$(nproc) \
    && make install \
    && cd /tmp && rm -rf protobuf-all-${PROTOBUF_VERSION}.tar.gz protobuf-${PROTOBUF_VERSION} \
    && git clone --depth 1 --recurse-submodules -b v1.46.0 https://github.com/grpc/grpc \
    && cd grpc \
    && mkdir -p cmake/build && cd cmake/build \
    && cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF ../.. \
    && make -j 4 \
    && make install \
    && cd /tmp && rm -rf grpc \
    && git clone --depth 1 https://github.com/google/benchmark.git \
    && git clone --depth 1 https://github.com/google/googletest.git benchmark/googletest \
    && cd benchmark \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make install \
    && cd /tmp && rm -rf benchmark \
    && ldconfig \
    && rm -rf /tmp/*

# Install high precision math libraries (MPFR, MPFRC++)
RUN cd /tmp \
    && echo "🔧 Installing high precision math libraries..." \
    && wget https://ftp.gnu.org/gnu/mpfr/mpfr-4.1.0.tar.gz \
    && tar -zxf mpfr-4.1.0.tar.gz \
    && cd mpfr-4.1.0 \
    && ./configure \
    && make -j$(nproc) \
    && make install \
    && cd /tmp && rm -rf mpfr-4.1.0 mpfr-4.1.0.tar.gz \
    && wget https://github.com/advanpix/mpreal/archive/refs/tags/mpfrc++-3.6.8.tar.gz \
    && tar -xzf mpfrc++-3.6.8.tar.gz \
    && cd mpreal-mpfrc-3.6.8 \
    && cp mpreal.h /usr/include/ \
    && cd /tmp && rm -rf mpreal-mpfrc-3.6.8 mpfrc++-3.6.8.tar.gz \
    && git clone --depth 1 https://github.com/pyomeca/ezc3d.git \
    && cd ezc3d \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd /tmp && rm -rf ezc3d \
    && ldconfig \
    && rm -rf /tmp/*
# Clone datasets and nimblephysics, build nimblephysics
WORKDIR /
RUN echo "🔧 Cloning datasets and building nimblephysics..." \
    && git -c http.sslVerify=false clone --depth 1 https://github.com/Rose-STL-Lab/UCSD-OpenCap-Fitness-Dataset.git \
    && git -c http.sslVerify=false clone --depth 1 https://github.com/shubhMaheshwari/nimblephysics.git \
    && cd /nimblephysics \
    && echo "🔧 Installing nimblephysics package..." \
    && pip install pybind11-stubgen \
    && python setup.py build_ext --inplace \
    && python setup.py install \
    && echo "✅ Nimblephysics package installed successfully!" \
    && echo "🔧 Testing nimblephysics installation..." \
    && python -c "\
import sys; \
sys.path.insert(0, '/nimblephysics/python'); \
import nimblephysics as nimble; \
print('✅ Nimblephysics imported successfully!'); \
print('📦 Complete physics stack: Eigen, CCD, ASSIMP, MUMPS, IPOPT, FCL, OSG, Protobuf, GRPC, MPFR, ezc3d'); \
print('🎯 Available components:', len([attr for attr in dir(nimble) if not attr.startswith('_')]), 'items')" \
    && echo "✅ Nimblephysics test completed successfully!" \
    && cd / \
    && rm -rf /UCSD-OpenCap-Fitness-Dataset/.git \
    && rm -rf /nimblephysics/.git \
    && find /nimblephysics/build -name "*.o" -delete \
    && find /nimblephysics/build -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true

# Clone main application and setup conda environment
WORKDIR /T2M-GPT
RUN git -c http.sslVerify=false clone --depth 1 https://gitlab.nrp-nautilus.io/shmaheshwari/digital-coach-anwesh.git . \
    && conda env create -f environment.yml \
    && conda clean -afy

# Set comprehensive environment variables
ENV PYTHONPATH="/nimblephysics/python:/nimblephysics/build/python:$PYTHONPATH"
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:/usr/local/envs/T2M-GPT/lib:$LD_LIBRARY_PATH"
ENV PKG_CONFIG_PATH="/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
ENV DISPLAY=:99.0

# Activate conda environment for subsequent commands
SHELL ["conda", "run", "-n", "T2M-GPT", "/bin/bash", "-c"]

# Download models and install Python packages
RUN bash dataset/prepare/download_model.sh \
    && bash dataset/prepare/download_extractor.sh \
    && pip install \
        ipykernel \
        deepspeed \
        polyscope \
        easydict \
        trimesh \
        networkx \
        tqdm \
        matplotlib \
        scipy \
        scikit-learn \
    && pip install --force-reinstall numpy==1.22.0 \
    && pip cache purge

# Test nimblephysics and create startup script
# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🌟 Digital Coach - Complete Physics Simulation Stack"\n\
echo "📦 Libraries: Eigen→CCD→ASSIMP→MUMPS→IPOPT→FCL→OSG→Protobuf→GRPC→MPFR→ezc3d→nimblephysics"\n\
echo "🖥️  Setting up virtual display..."\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
sleep 2\n\
echo "🐍 Initializing conda environment..."\n\
source /usr/local/etc/profile.d/conda.sh > /dev/null 2>&1 || true\n\
conda init bash > /dev/null 2>&1 || true\n\
source ~/.bashrc > /dev/null 2>&1 || true\n\
echo "✅ Environment ready"\n\
echo "🚀 Executing: $@"\n\
exec "$@"' > /start-digital-coach.sh \
    && chmod +x /start-digital-coach.sh

ENTRYPOINT ["/start-digital-coach.sh"]
CMD ["conda", "run", "--no-capture-output", "-n", "T2M-GPT", "bash"]