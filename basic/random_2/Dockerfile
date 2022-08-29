ARG PYTHON_VERSION=3.7
FROM python:$PYTHON_VERSION

RUN apt-get -qy update && \
  apt-get -qy install libgl1 && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "./agent.py" ]
