.PHONY: generate build

generate:
	go generate ./...

build: generate
	go build -o bin/syscall-collector .
