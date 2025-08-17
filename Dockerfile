# This file is for use as a devcontainer and a runtime container
#
# The devcontainer should use the build target and run as root with podman
# or docker with user namespaces.
#
FROM python:3.10 AS build

# Create working directory
WORKDIR /workspace
COPY . /workspace
# set up a virtual environment and put it in PATH
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH

# install python package into /venv
RUN pip install --upgrade pip
RUN pip install .


# Add apt-get system dependencies for runtime here if needed
# RUN apt-get update && apt-get upgrade -y && \
#     apt-get install -y --no-install-recommends \
#     desired-packages \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install libgl1 -y
RUN apt-get install libglx-mesa0 -y

# ENTRYPOINT ["uvicorn", "main:app", "--host", "172.23.169.93", "--port", "8000"]

EXPOSE 8000
