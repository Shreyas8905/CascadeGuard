"""
Loader for the real ADFA-LD dataset (Creech & Hu, 2013) --
https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset

Real syscall-number traces collected on a live, patched Ubuntu 11.04 host
running MySQL/FTP/SSH, with six genuine attack categories: Adduser,
Hydra_FTP, Hydra_SSH, Java_Meterpreter, Meterpreter, Web_Shell.

Key structural difference from the earlier synthetic generator: ADFA-LD has
no pod/namespace/label metadata to group entities the way ShadowKube groups
pods. There is therefore no honest way to build multiple per-"scenario"
baselines the way the synthetic version did -- we use ONE global baseline
built from the real Training_Data_Master traces (833 normal sequences),
scored against by both the real Validation_Data_Master traces (4372 normal
sequences, used as held-out normal) and the six real attack categories. The
attack category is retained as a field for stratified reporting only, never
for baseline grouping -- this is intentionally the harder, more honest
setting (ShadowKube's own grouping-by-metadata is itself a design choice
that assumes homogeneous within-group behavior; here we test without that
assumption).
"""
import glob
import os
import re

DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "adfa_ld", "extracted", "ADFA-LD")

ATTACK_CATEGORIES = ["Adduser", "Hydra_FTP", "Hydra_SSH",
                      "Java_Meterpreter", "Meterpreter", "Web_Shell"]

GLOBAL_GROUP = "host"  # single baseline group, see module docstring


def _read_trace(path):
    with open(path) as f:
        return f.read().split()


def load_syscall_names(root=DEFAULT_ROOT):
    syscall_list_path = os.path.join(os.path.dirname(root), "..", "ADFA-LD+Syscall+List.txt")
    syscall_list_path = os.path.normpath(syscall_list_path)
    if not os.path.exists(syscall_list_path):
        return {}
    text = open(syscall_list_path, encoding="utf-8", errors="ignore").read()
    pairs = re.findall(r"#define __NR_(\w+)\s+(\d+)", text)
    return {num: name for name, num in pairs}


def load_adfa(root=DEFAULT_ROOT):
    """Returns (train_normal, test_normal, attack_entities) where each is a
    list of entity dicts: {"trace_id", "scenario" (== GLOBAL_GROUP always),
    "attack_category" (None for normal), "true", "seq"}."""
    train_files = sorted(glob.glob(os.path.join(root, "Training_Data_Master", "*.txt")))
    val_files = sorted(glob.glob(os.path.join(root, "Validation_Data_Master", "*.txt")))

    train_normal = [{"trace_id": os.path.basename(p), "scenario": GLOBAL_GROUP,
                      "attack_category": None, "true": "normal", "seq": _read_trace(p)}
                     for p in train_files]
    test_normal = [{"trace_id": os.path.basename(p), "scenario": GLOBAL_GROUP,
                     "attack_category": None, "true": "normal", "seq": _read_trace(p)}
                    for p in val_files]

    attack_entities = []
    for cat in ATTACK_CATEGORIES:
        cat_files = sorted(glob.glob(os.path.join(root, "Attack_Data_Master", f"{cat}_*", "*.txt")))
        for p in cat_files:
            attack_entities.append({
                "trace_id": os.path.basename(p), "scenario": GLOBAL_GROUP,
                "attack_category": cat, "true": "attack", "seq": _read_trace(p),
            })
    return train_normal, test_normal, attack_entities


if __name__ == "__main__":
    train_normal, test_normal, attack_entities = load_adfa()
    print(f"train_normal={len(train_normal)} test_normal={len(test_normal)} "
          f"attack={len(attack_entities)}")
    from collections import Counter
    print(Counter(e["attack_category"] for e in attack_entities))
    names = load_syscall_names()
    print(f"parsed {len(names)} syscall-number-to-name mappings, "
          f"e.g. 6 -> {names.get('6')}")
