# Use an official Ubuntu runtime as a parent image
FROM ubuntu:25.04

# Set the working directory
WORKDIR /BIGE

# Install necessary dependencies
RUN apt-get update && apt-get install -y wget git htop xvfb build-essential 

# Install Miniconda
RUN MINICONDA_INSTALLER_SCRIPT=Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    MINICONDA_PREFIX=/usr/local && \
    wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT && \
    chmod +x $MINICONDA_INSTALLER_SCRIPT && \
    ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX && \
    rm $MINICONDA_INSTALLER_SCRIPT

# Update PATH to include conda
ENV PATH=/usr/local/bin:$PATH

# Copy the entire repo (including environment.yml) into the image
COPY . /BIGE

# Create the conda environment from environment.yml
RUN conda env create -f environment.yaml

# Activate the conda environment for subsequent RUN commands
SHELL ["conda", "run", "-n", "BIGE", "/bin/bash", "-c"]

# # Install additional Python packages (if needed)
# RUN pip install --user ipykernel polyscope easydict trimesh
# RUN pip install --user --force-reinstall numpy==1.22.0

# # Set up Xvfb for Polyscope
# ENV DISPLAY=:99.0
# RUN Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Set the entrypoint to use the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "BIGE", "python"]