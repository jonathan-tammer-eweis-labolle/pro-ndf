FROM nvcr.io/nvidia/pytorch:23.07-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

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

RUN npm install -g @anthropic-ai/claude-code

RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Register the Jupyter kernel
RUN python -m ipykernel install --name=prondf --display-name="ProNDF"

ENV PYTHONUNBUFFERED=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

CMD ["/bin/bash"]
