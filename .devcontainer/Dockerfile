# Start from NVIDIA's PyTorch image (includes CUDA, cuDNN, PyTorch)
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    pytorch-lightning \
    dill \
    jupyter \
    ipykernel \
    black \

# Install ipykernel for Jupyter
RUN python -m ipykernel install --name=anthropic_test --display-name="Anthropic Test"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Default command
CMD ["/bin/bash"]
