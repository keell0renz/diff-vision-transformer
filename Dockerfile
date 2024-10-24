# Use the latest CUDA image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

WORKDIR /diff-vision-transformer

COPY . .

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN echo 'cd /diff-vision-transformer' >> ~/.bashrc && \
    echo 'source .venv/bin/activate' >> ~/.bashrc

RUN uv venv

ENV UV_HTTP_TIMEOUT=60

RUN uv sync

CMD ["/bin/bash"]
