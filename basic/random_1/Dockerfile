ARG FROM=ghcr.io/diambra/arena-base-on3.10-bullseye:main
FROM $FROM

WORKDIR /app
COPY . .

ENTRYPOINT [ "./agent.py" ]
