#!/usr/bin/env python3
"""
Simple traffic generator helper for Mininet.
Generates a JSON schedule and a shell script with Mininet `host iperf` commands
that you can run inside the Mininet host or paste into the Mininet CLI.

Usage examples:
  # generate a schedule and commands (no execution)
  python tools/traffic_gen.py --pairs h1:h2,h2:h3 --duration 20 --bw 10M --out traffic_schedule.json

  # execute locally (ONLY if running on the same machine as Mininet and on Linux)
  python tools/traffic_gen.py --pairs h1:h2 --duration 10 --bw 5M --execute

Notes:
- The script does not interact with Mininet programmatically. It produces commands
  of the form `h1 iperf -s &` and `h2 iperf -c h1 -t <dur> -b <bw> &` which can be
  run in the Mininet CLI or in a shell where Mininet hosts are available (e.g., inside
  a Mininet VM using `mnexec`).
- Use `--start-offset` to stagger flows.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Mininet iperf traffic commands')
    parser.add_argument('--pairs', type=str, required=True,
                        help='Comma-separated host pairs src:dst (e.g. h1:h2,h2:h3)')
    parser.add_argument('--duration', type=int, default=10, help='Duration for each iperf client (seconds)')
    parser.add_argument('--bw', type=str, default='10M', help='Bandwidth argument to iperf (e.g. 10M, 100M)')
    parser.add_argument('--start-offset', type=float, default=0.5, help='Seconds between starting successive flows')
    parser.add_argument('--out', type=str, default='traffic_schedule.json', help='Output JSON schedule path')
    parser.add_argument('--script', type=str, default='traffic_commands.sh', help='Output shell script path')
    parser.add_argument('--execute', action='store_true', help='Attempt to execute commands locally (Linux only)')
    return parser.parse_args()


def build_schedule(pairs, duration, bw, start_offset):
    schedule = []
    t = 0.0
    for pair in pairs:
        src, dst = pair.split(':')
        entry = {
            'src': src,
            'dst': dst,
            'start': round(t, 3),
            'duration': duration,
            'bw': bw
        }
        schedule.append(entry)
        t += start_offset
    return schedule


def make_scripts(schedule, script_path):
    lines = ["#!/bin/bash\n"]
    # start servers first
    servers = set([item['src'] for item in schedule])
    # Actually any host that will be a server (dst) should start iperf -s
    servers = set([item['dst'] for item in schedule])
    for s in servers:
        lines.append(f"# start iperf server on {s}\n")
        lines.append(f"{s} iperf -s -u > /tmp/{s}_iperf_server.log 2>&1 &\n")
    lines.append('\n')

    for item in schedule:
        # sleep till start
        if item['start'] > 0:
            lines.append(f"sleep {item['start']}\n")
        src = item['src']
        dst = item['dst']
        dur = item['duration']
        bw = item['bw']
        # use UDP iperf by default for consistency; change to -u flag
        lines.append(f"# start client {src} -> {dst}\n")
        lines.append(f"{src} iperf -c {dst} -u -t {dur} -b {bw} > /tmp/{src}_to_{dst}_iperf_client.log 2>&1 &\n")
        lines.append('\n')
    # add a final wait
    lines.append("wait\n")

    with open(script_path, 'w') as fh:
        fh.writelines(lines)
    os.chmod(script_path, 0o755)
    return script_path


def execute_schedule(schedule):
    # Execute commands locally using subprocess; assumes Mininet CLI host names are available
    # This is only safe if running inside the Mininet host environment (e.g., a Mininet VM)
    if sys.platform.startswith('win'):
        print('Local execution is not supported on Windows. Generate scripts instead.')
        return
    # Start servers
    servers = set([item['dst'] for item in schedule])
    procs = []
    for s in servers:
        cmd = f"{s} iperf -s -u"
        print('Running:', cmd)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(p)
    # Start clients with delays
    for item in schedule:
        start = item['start']
        if start > 0:
            time.sleep(start)
        src = item['src']
        dst = item['dst']
        dur = item['duration']
        bw = item['bw']
        cmd = f"{src} iperf -c {dst} -u -t {dur} -b {bw}"
        print('Running:', cmd)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(p)
    # Wait for processes
    for p in procs:
        try:
            p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    print('Execution attempted. Check Mininet or /tmp/*_iperf*.log for outputs')


def main():
    args = parse_args()
    pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
    schedule = build_schedule(pairs, args.duration, args.bw, args.start_offset)
    # save schedule
    with open(args.out, 'w') as fh:
        json.dump(schedule, fh, indent=2)
    print(f'Wrote schedule to {args.out} ({len(schedule)} flows)')

    # create script
    script_path = make_scripts(schedule, args.script)
    print(f'Wrote Mininet commands script to {script_path}')

    if args.execute:
        print('Attempting local execution (only for Linux/Mininet environments)...')
        execute_schedule(schedule)
    else:
        print('To run the traffic:')
        print(' - Open Mininet CLI and paste the commands from', script_path)
        print(' - Or run the script inside the Mininet host environment (Linux) using:')
        print('     sudo bash', script_path)


if __name__ == '__main__':
    main()
