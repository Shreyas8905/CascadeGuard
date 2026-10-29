package main

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang -cflags "-I/usr/include/x86_64-linux-gnu" syscalls bpf/syscalls.bpf.c
