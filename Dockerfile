FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Update and install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as the default 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Verify installation
RUN python --version && pip --version