package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
)

const (
	phaseEnter = 1
	phaseExit  = 2
)

type kernelEvent struct {
	TimestampNS uint64
	CgroupID    uint64
	PidTgid     uint64
	UidGid      uint64
	SyscallID   int64
	ReturnValue int64
	CPU         uint32
	Phase       uint32
	Comm        [16]byte
}
type event struct {
	SchemaVersion string `json:"schema_version"`
	HostID        string `json:"host_id"`
	TimestampNS   uint64 `json:"timestamp_ns"`
	CgroupID      uint64 `json:"cgroup_id"`
	PID           uint32 `json:"pid"`
	TID           uint32 `json:"tid"`
	UID           uint32 `json:"uid"`
	GID           uint32 `json:"gid"`
	SyscallID     int64  `json:"syscall_id"`
	Phase         string `json:"phase"`
	ReturnValue   int64  `json:"return_value"`
	CPU           uint32 `json:"cpu"`
	Comm          string `json:"comm"`
}
type batch struct {
	SchemaVersion string  `json:"schema_version"`
	HostID        string  `json:"host_id"`
	Events        []event `json:"events"`
}

func main() {
	endpoint := flag.String("endpoint", os.Getenv("IDS_URL"), "optional batch endpoint; empty writes JSONL to stdout")
	token := flag.String("token", os.Getenv("INGEST_TOKEN"), "optional ingestion token")
	hostID := flag.String("host-id", envOr("HOSTNAME", "unknown-host"), "stable node identifier")
	batchSize := flag.Int("batch-size", 256, "maximum events per HTTP batch")
	flushEvery := flag.Duration("flush-interval", 250*time.Millisecond, "maximum batch flush interval")
	flag.Parse()
	if *batchSize < 1 || *batchSize > 10000 {
		log.Fatal("batch-size must be between 1 and 10000")
	}
	if *flushEvery <= 0 {
		log.Fatal("flush-interval must be positive")
	}
	if err := run(*endpoint, *token, *hostID, *batchSize, *flushEvery); err != nil {
		log.Fatal(err)
	}
}
func run(endpoint, token, hostID string, batchSize int, flushEvery time.Duration) error {
	spec, err := loadSyscalls()
	if err != nil {
		return fmt.Errorf("load eBPF object (run go generate ./... first): %w", err)
	}
	objs := syscallsObjects{}
	if err := spec.LoadAndAssign(&objs, nil); err != nil {
		return fmt.Errorf("load eBPF programs: %w", err)
	}
	defer objs.Close()
	lEnter, err := link.Tracepoint("raw_syscalls", "sys_enter", objs.OnSysEnter, nil)
	if err != nil {
		return fmt.Errorf("attach sys_enter: %w", err)
	}
	defer lEnter.Close()
	lExit, err := link.Tracepoint("raw_syscalls", "sys_exit", objs.OnSysExit, nil)
	if err != nil {
		return fmt.Errorf("attach sys_exit: %w", err)
	}
	defer lExit.Close()
	reader, err := ringbuf.NewReader(objs.Events)
	if err != nil {
		return fmt.Errorf("open ring buffer: %w", err)
	}
	defer reader.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	log.Printf("syscall tracepoints attached host_id=%s endpoint=%s", hostID, endpoint)
	out := make(chan event, batchSize*4)
	readErr := make(chan error, 1)
	go func() {
		for {
			rec, err := reader.Read()
			if err != nil {
				readErr <- err
				return
			}
			var raw kernelEvent
			if err := binary.Read(bytes.NewReader(rec.RawSample), binary.NativeEndian, &raw); err != nil {
				log.Printf("decode event: %v", err)
				continue
			}
			phase := "unknown"
			if raw.Phase == phaseEnter {
				phase = "enter"
			}
			if raw.Phase == phaseExit {
				phase = "exit"
			}
			ev := event{SchemaVersion: "1.0", HostID: hostID, TimestampNS: raw.TimestampNS, CgroupID: raw.CgroupID,
				PID: uint32(raw.PidTgid >> 32), TID: uint32(raw.PidTgid), UID: uint32(raw.UidGid), GID: uint32(raw.UidGid >> 32),
				SyscallID: raw.SyscallID, Phase: phase, ReturnValue: raw.ReturnValue, CPU: raw.CPU, Comm: strings.TrimRight(string(raw.Comm[:]), "\x00")}
			select {
			case out <- ev:
			case <-ctx.Done():
				return
			}
		}
	}()
	client := &http.Client{Timeout: 5 * time.Second}
	ticker := time.NewTicker(flushEvery)
	defer ticker.Stop()
	pending := make([]event, 0, batchSize)
	flush := func() error {
		if len(pending) == 0 {
			return nil
		}
		if endpoint == "" {
			enc := json.NewEncoder(os.Stdout)
			for _, ev := range pending {
				if err := enc.Encode(ev); err != nil {
					return err
				}
			}
		} else {
			body, err := json.Marshal(batch{SchemaVersion: "1.0", HostID: hostID, Events: pending})
			if err != nil {
				return err
			}
			req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
			if err != nil {
				return err
			}
			req.Header.Set("Content-Type", "application/json")
			if token != "" {
				req.Header.Set("X-Ingest-Token", token)
			}
			resp, err := client.Do(req)
			if err != nil {
				return err
			}
			resp.Body.Close()
			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				return fmt.Errorf("endpoint returned HTTP %d", resp.StatusCode)
			}
		}
		pending = pending[:0]
		return nil
	}
	for {
		select {
		case <-ctx.Done():
			_ = flush()
			return nil
		case err := <-readErr:
			return fmt.Errorf("ring buffer read: %w", err)
		case ev := <-out:
			pending = append(pending, ev)
			if len(pending) >= batchSize {
				if err := flush(); err != nil {
					log.Printf("flush failed; retaining batch: %v", err)
				}
			}
		case <-ticker.C:
			if err := flush(); err != nil {
				log.Printf("flush failed; retaining batch: %v", err)
			}
		}
	}
}
func envOr(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}
