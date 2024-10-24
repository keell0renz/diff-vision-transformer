# Use the latest CUDA image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /diff-vision-transformer

# Copy the current directory contents into the container at /app
COPY . .

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN echo 'cd /diff-vision-transformer' >> ~/.bashrc && \
    echo 'source .venv/bin/activate' >> ~/.bashrc

# Create a virtual environment
RUN uv venv

ENV UV_HTTP_TIMEOUT=60

# Install python dependencies
RUN uv sync

CMD ["/bin/bash"]
