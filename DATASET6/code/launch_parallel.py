#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行启动器：按分片并行运行 macro_py.py
- 默认适配 --backend imagej：每个进程内部强制单线程；通过多进程分片并行
- 其它后端（skimage/morph）可自定义每进程线程数
- 为每个分片写独立日志文件，便于查看进度

示例（Windows, 从项目根目录执行）：
  python DATASET7\code\launch_parallel.py \
    -i DATASET7\data\Day1\images \
    -o DATASET7\data\Day1\processed \
    -r 10 --backend imagej --shards 4

按 Ctrl+C 可一键终止所有子进程。
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import multiprocessing as mp
import sys
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel launcher for macro_py.py with sharding")
    parser.add_argument("-i", "--input", required=True, help="Input directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-r", "--rolling", type=int, default=10, help="Rolling radius (pixels), like ImageJ rolling=10")
    parser.add_argument("--backend", choices=["imagej", "skimage", "morph"], default="imagej", help="Background subtraction backend")
    parser.add_argument("--shards", type=int, default=0, help="Total number of shards/processes (0=auto: CPU count or 4, whichever smaller >=2)")
    parser.add_argument("--per-workers", default=None, help="Workers per process: 'auto' or int. For imagej this is forced to 1.")
    parser.add_argument("--logs", default=None, help="Directory to write logs (default: <output>/logs_parallel)")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    macro_py = (Path(__file__).parent / "macro_py.py").resolve()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if args.shards <= 0:
        cpu = max(2, (mp.cpu_count() or 4))
        args.shards = min(cpu, 4)  # 保守默认：最多 4 个并行进程

    logs_dir = Path(args.logs) if args.logs else (output_dir / "logs_parallel")
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "imagej":
        per_workers = "1"
    else:
        per_workers = str(args.per_workers) if args.per_workers else "auto"

    commands: list[list[str]] = []
    for shard_index in range(args.shards):
        cmd = [
            sys.executable,
            str(macro_py),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-r", str(args.rolling),
            "--backend", args.backend,
            "--workers", per_workers,
            "--num-shards", str(args.shards),
            "--shard-index", str(shard_index),
        ]
        commands.append(cmd)

    print(f"项目根目录: {project_root}")
    print(f"脚本: {macro_py}")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print(f"后端: {args.backend} | 分片: {args.shards} | 每进程workers: {per_workers}")
    print(f"日志目录: {logs_dir}")

    for idx, cmd in enumerate(commands):
        print(f"[dry-run={args.dry_run}] shard {idx+1}/{args.shards}: {' '.join(cmd)}")

    if args.dry_run:
        return

    procs: list[subprocess.Popen] = []
    try:
        for idx, cmd in enumerate(commands):
            log_path = logs_dir / f"shard_{idx}.log"
            log_f = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                creationflags=0 if os.name != 'nt' else 0,  # 使用当前窗口
            )
            procs.append(proc)
            print(f"已启动 shard {idx+1}/{args.shards} | PID={proc.pid} | 日志={log_path}")

        # 等待所有进程完成
        exit_codes = []
        for idx, proc in enumerate(procs):
            code = proc.wait()
            exit_codes.append(code)
            print(f"分片 {idx+1}/{args.shards} 结束，退出码={code}")

        all_ok = all(code == 0 for code in exit_codes)
        print("全部完成" if all_ok else f"部分失败: {exit_codes}")
        if not all_ok:
            sys.exit(1)
    except KeyboardInterrupt:
        print("收到中断信号，正在终止所有子进程…")
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        print("已请求终止。可查看日志了解最后输出。")
        sys.exit(1)


if __name__ == "__main__":
    main()
