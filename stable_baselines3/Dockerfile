FROM ghcr.io/diambra/arena-stable-baselines3-on3.10-bullseye:fix-sb3-seed

RUN pip install huggingface_hub
ENV HF_HOME=/tmp
WORKDIR /app
COPY . .

ENTRYPOINT [ "python", "./agent.py" ]