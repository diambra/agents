ARG TAG=main
FROM ghcr.io/diambra/arena-base-on3.10-bullseye:$TAG

WORKDIR /app
COPY . .

ENTRYPOINT [ "./agent.py" ]
