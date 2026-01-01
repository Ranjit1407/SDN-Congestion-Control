#!/usr/bin/env python3
# tools/parse_logs.py
#
# Robust parser for logs/stats_log.csv -> data/features.csv
# - Handles 8-column Ryu port stats (timestamp, dpid, port_no, rx_bytes, tx_bytes, rx_packets, tx_packets, duration_sec)
# - Skips malformed rows
# - Computes per-interval deltas/rates per (dpid,port_no)
# - Outputs: timestamp, dpid, port_no, tx_bps, rx_bps, tx_pps, rx_pps

import os
import csv
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_PATH = os.path.join(PROJECT_ROOT, 'logs', 'stats_log.csv')
OUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'features.csv')

REQUIRED_NUMERIC = [
    'timestamp', 'dpid', 'port_no',
    'rx_bytes', 'tx_bytes', 'rx_packets', 'tx_packets'
    # duration_sec is optional
]

def read_stats(path):
    rows = []
    if not os.path.exists(path):
        print(f"[!] Log file not found: {path}")
        return rows

    with open(path, 'r', newline='') as f:
        # Use DictReader to be resilient to column order
        r = csv.DictReader(f)
        if r.fieldnames is None:
            print("[!] CSV appears empty or missing header.")
            return rows

        # normalize headers (strip spaces)
        headers = [h.strip() for h in r.fieldnames]
        # quick sanity
        missing = [h for h in REQUIRED_NUMERIC if h not in headers]
        if missing:
            print(f"[!] CSV missing expected columns: {missing}")
            print(f"    Found columns: {headers}")
            # continue anyway; we'll try to parse what we can

        for i, row in enumerate(r, start=2):  # start=2 (line after header)
            try:
                rec = {
                    'timestamp': float(row.get('timestamp', '').strip() or 0.0),
                    'dpid': int(row.get('dpid', '').strip() or 0),
                    'port_no': int(row.get('port_no', '').strip() or -1),
                    'rx_bytes': int(row.get('rx_bytes', '').strip() or 0),
                    'tx_bytes': int(row.get('tx_bytes', '').strip() or 0),
                    'rx_packets': int(row.get('rx_packets', '').strip() or 0),
                    'tx_packets': int(row.get('tx_packets', '').strip() or 0),
                    'duration_sec': int((row.get('duration_sec') or '0').strip() or 0),
                }
                # basic sanity
                if rec['port_no'] < 0 or rec['timestamp'] <= 0:
                    continue
                rows.append(rec)
            except Exception as e:
                # Skip malformed lines silently but informative
                # print(f"[skip line {i}] {e}")
                continue
    return rows

def compute_features(rows):
    # Group by (dpid, port_no) and sort by time
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dpid'], r['port_no'])].append(r)

    feats = []
    for key, grp in groups.items():
        grp.sort(key=lambda x: x['timestamp'])
        prev = None
        for cur in grp:
            if prev is None:
                prev = cur
                continue
            dt = cur['timestamp'] - prev['timestamp']
            if dt <= 0:
                prev = cur
                continue

            d_tx_bytes = cur['tx_bytes'] - prev['tx_bytes']
            d_rx_bytes = cur['rx_bytes'] - prev['rx_bytes']
            d_tx_pkts  = cur['tx_packets'] - prev['tx_packets']
            d_rx_pkts  = cur['rx_packets'] - prev['rx_packets']

            # guard against counter resets/wraps
            if d_tx_bytes < 0 or d_rx_bytes < 0 or d_tx_pkts < 0 or d_rx_pkts < 0:
                prev = cur
                continue

            tx_bps = (d_tx_bytes * 8.0) / dt
            rx_bps = (d_rx_bytes * 8.0) / dt
            tx_pps = d_tx_pkts / dt
            rx_pps = d_rx_pkts / dt

            feats.append({
                'timestamp': cur['timestamp'],
                'dpid': cur['dpid'],
                'port_no': cur['port_no'],
                'tx_bps': tx_bps,
                'rx_bps': rx_bps,
                'tx_pps': tx_pps,
                'rx_pps': rx_pps
            })
            prev = cur

    feats.sort(key=lambda x: (x['timestamp'], x['dpid'], x['port_no']))
    return feats

def write_features(path, feats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp','dpid','port_no','tx_bps','rx_bps','tx_pps','rx_pps'])
        for r in feats:
            w.writerow([
                f"{r['timestamp']:.6f}", r['dpid'], r['port_no'],
                f"{r['tx_bps']:.6f}", f"{r['rx_bps']:.6f}",
                f"{r['tx_pps']:.6f}", f"{r['rx_pps']:.6f}"
            ])

def main():
    rows = read_stats(LOG_PATH)
    if not rows:
        print(f"[!] No usable rows in {LOG_PATH}")
        return
    feats = compute_features(rows)
    if not feats:
        print("[!] No feature rows produced (need >=2 samples per port).")
        return
    write_features(OUT_PATH, feats)
    print(f"[+] Wrote {len(feats)} feature rows to {OUT_PATH}")

if __name__ == "__main__":
    main()

