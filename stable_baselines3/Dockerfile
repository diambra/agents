ARG TAG=main
FROM ghcr.io/diambra/arena-stable-baselines3-on3.10-bullseye:$TAG

RUN pip install huggingface_hub
ENV HF_HOME=/tmp
WORKDIR /app
COPY . .

ENTRYPOINT [ "python", "./agent.py" ]