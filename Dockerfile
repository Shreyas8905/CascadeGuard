FROM golang:1.23-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends clang llvm libbpf-dev linux-libc-dev make && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN go generate ./... && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/syscall-collector .
FROM debian:bookworm-slim
COPY --from=build /out/syscall-collector /usr/local/bin/syscall-collector
USER 0
ENTRYPOINT ["/usr/local/bin/syscall-collector"]
