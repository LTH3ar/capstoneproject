FROM mcr.microsoft.com/devcontainers/python:3.10

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends build-essential gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install jupyterlab
RUN pip install jellyfish sentence-transformers scikit-learn