ARG PYTHON_VERSION=3.7
FROM python:$PYTHON_VERSION

RUN apt-get -qy update && \
  apt-get -qy install libgl1 cmake libopenmpi-dev python3-dev zlib1g-dev && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "python3", "agent.py", "--cfgFile", "/app/cfg_files/doapp/sr6_128x4_das_nc.yaml", "--trainedModel", "25M"]
