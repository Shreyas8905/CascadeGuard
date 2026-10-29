// SPDX-License-Identifier: MIT
#include <linux/bpf.h>
#include <linux/types.h>
#include <bpf/bpf_helpers.h>

char LICENSE[] SEC("license") = "Dual MIT/GPL";
#define TASK_COMM_LEN 16
#define PHASE_ENTER 1
#define PHASE_EXIT 2

struct trace_entry { __u16 type; __u8 flags; __u8 preempt_count; __s32 pid; };
struct trace_event_raw_sys_enter { struct trace_entry ent; long id; unsigned long args[6]; };
struct trace_event_raw_sys_exit { struct trace_entry ent; long id; long ret; };
struct syscall_event {
    __u64 timestamp_ns, cgroup_id, pid_tgid, uid_gid;
    __s64 syscall_id, ret;
    __u32 cpu, phase;
    char comm[TASK_COMM_LEN];
};
struct { __uint(type, BPF_MAP_TYPE_RINGBUF); __uint(max_entries, 1 << 24); } events SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, __u64); } dropped SEC(".maps");
static __always_inline void count_drop(void) {
    __u32 key = 0; __u64 *v = bpf_map_lookup_elem(&dropped, &key); if (v) __sync_fetch_and_add(v, 1);
}
static __always_inline int emit_event(__s64 id, __s64 ret, __u32 phase) {
    struct syscall_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) { count_drop(); return 0; }
    e->timestamp_ns = bpf_ktime_get_ns(); e->cgroup_id = bpf_get_current_cgroup_id();
    e->pid_tgid = bpf_get_current_pid_tgid(); e->uid_gid = bpf_get_current_uid_gid();
    e->syscall_id = id; e->ret = ret; e->cpu = bpf_get_smp_processor_id(); e->phase = phase;
    bpf_get_current_comm(&e->comm, sizeof(e->comm)); bpf_ringbuf_submit(e, 0); return 0;
}
SEC("tracepoint/raw_syscalls/sys_enter") int on_sys_enter(struct trace_event_raw_sys_enter *ctx) { return emit_event(ctx->id, 0, PHASE_ENTER); }
SEC("tracepoint/raw_syscalls/sys_exit") int on_sys_exit(struct trace_event_raw_sys_exit *ctx) { return emit_event(ctx->id, ctx->ret, PHASE_EXIT); }
