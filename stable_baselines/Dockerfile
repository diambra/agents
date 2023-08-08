ARG TAG=main
FROM ghcr.io/diambra/arena-stable-baselines-on3.7-bullseye:$TAG

WORKDIR /app
COPY . .

ENTRYPOINT [ "python3", "agent.py", "--cfgFile", "/app/cfg_files/doapp/sr6_128x4_das_nc.yaml", "--trainedModel", "model"]
