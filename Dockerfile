
# This file is for use as a devcontainer and a runtime container
#
# The devcontainer should use the build target and run as root with podman
# or docker with user namespaces.
#

# Build static UI
FROM node:20 AS ui-build
WORKDIR /workspace/calibration_ui
COPY calibration_ui/ .
RUN npm install
RUN npm run build

FROM python:3.10 AS build

# Create working directory
WORKDIR /workspace
COPY . /workspace
# set up a virtual environment and put it in PATH
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH

# install python package into /venv
RUN pip install --upgrade pip
RUN pip install .[dev]

RUN apt-get update
RUN apt-get install libgl1 -y
RUN apt-get install libglx-mesa0 -y

COPY --from=ui-build /workspace/calibration_ui/dist ./static

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000
