# Use an official Ubuntu runtime as a parent image
FROM ubuntu:22.04

# Set the working directory
WORKDIR /T2M-GPT

# Install necessary dependencies
RUN apt-get update && apt-get install -y wget git htop

# Install Miniconda
RUN MINICONDA_INSTALLER_SCRIPT=Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    MINICONDA_PREFIX=/usr/local && \
    wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT && \
    chmod +x $MINICONDA_INSTALLER_SCRIPT && \
    ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX && \
    rm $MINICONDA_INSTALLER_SCRIPT

# Update PATH to include conda
ENV PATH=/usr/local/bin:$PATH

# Clone the digital-coach-anwesh repository
RUN git -c http.sslVerify=false clone https://gitlab.nrp-nautilus.io/shmaheshwari/digital-coach-anwesh.git .

# Copy the environment.yml file and create the conda environment
# COPY digital-coach-anwesh/environment.yml /T2M-GPT/environment.yml
RUN conda env create -f environment.yml

# Activate the conda environment
SHELL ["conda", "run", "-n", "T2M-GPT", "/bin/bash", "-c"]

# Download the model and extractor
RUN bash dataset/prepare/download_model.sh && \
    bash dataset/prepare/download_extractor.sh

# Install additional Python packages
RUN pip install --user ipykernel nimblephysics deepspeed polyscope

# Install CUDA toolkit
# RUN apt-get install -y cuda-toolkit-11-2

# Set up Xvfb for Polyscope
RUN apt-get install -y xvfb
ENV DISPLAY=:99.0

# Create a fake screen
RUN Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Expose ports 443 and 80
# EXPOSE 443
# EXPOSE 80

# Set the entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "T2M-GPT", "python"]