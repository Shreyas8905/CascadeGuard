# Syscall Collector — Milestone 1

Linux eBPF syscall entry/exit capture using C eBPF and a Go userspace reader. JSON Lines are written to stdout by default; set `IDS_URL` to POST JSON batches to a compatible endpoint. This is an initial development scaffold, not production-ready.

## Captured fields

Schema version, host ID, monotonic timestamp, cgroup ID, PID/TID, UID/GID, syscall number, entry/exit phase, exit return value, CPU, and process `comm`. It does not copy syscall argument payloads. IDs are architecture-specific and must be mapped to the training vocabulary before Cascade-Guard inference.

## Build on Linux

Requires Go 1.23+, clang/LLVM, libbpf headers, and Linux UAPI headers. The Cilium eBPF guide: https://ebpf-go.dev/guides/getting-started/

```bash
sudo apt-get update
sudo apt-get install -y golang clang llvm libbpf-dev linux-libc-dev make
go mod tidy
go generate ./...
go build -o syscall-collector .
sudo ./syscall-collector
```

For HTTP batching, set `IDS_URL` and optionally `INGEST_TOKEN`. The endpoint must accept `POST` JSON `{ "schema_version":"1.0", "host_id":"...", "events":[...] }` and header `X-Ingest-Token`. The existing dashboard starter does not yet implement this batch contract.

## Helm deployment

Build and push the image to a registry accessible to the cluster. Edit `deploy/helm/syscall-collector/values.yaml`, then:

```bash
helm upgrade --install syscall-collector ./deploy/helm/syscall-collector \
  --namespace ids-system --create-namespace \
  --set image.repository=REGISTRY/PROJECT/syscall-collector \
  --set image.tag=0.1.0
```

For a remote endpoint, set `endpoint` and configure `existingTokenSecret` after creating a Kubernetes Secret. The DaemonSet targets Linux nodes only.

## Known limitations for this milestone

- Ring-buffer reservation failures are counted in a BPF map but not yet exported to userspace metrics.
- Container/pod metadata enrichment is not implemented yet; cgroup ID is emitted as a correlation key.
- HTTP retry/backoff and durable buffering are not implemented. Do not point this version at a production endpoint.
- The endpoint contract is a batch contract, not the single-event endpoint in the earlier dashboard starter.
- Kernel privileges vary. This chart requests `BPF`, `PERFMON`, and `SYS_RESOURCE`; some kernels or cluster policies may require different permissions or reject attachment. Test on a disposable Linux node first.
- Tracepoint coverage does not guarantee zero event loss under overload. Entry and exit are distinct records; missing exit records can occur when a process exits or events are lost.

## Next milestones

1. Export drop counters and add health metrics.
2. Add bounded queueing, retry/backoff, TLS validation, and acknowledged delivery.
3. Enrich cgroup IDs with container/pod metadata.
4. Add batch ingestion to the API and event store.
5. Integrate persisted Tier 3 model artifacts and exact training-time feature construction.
6. Benchmark event loss and node overhead under syscall-heavy workloads.
