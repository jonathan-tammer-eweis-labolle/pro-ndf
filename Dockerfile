# Base image: PyTorch with CUDA + cuDNN (change cuda version to match your driver)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Install Node.js 20 LTS, git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Pre-install Python dependencies so the layer is cached
# torch is already in the base image; install everything else
WORKDIR /workspace
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        numpy \
        scipy \
        matplotlib \
        pytorch-lightning \
        dill

# Entrypoint: editable-installs the project, then execs the command
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["claude", "--dangerously-skip-permissions"]
