#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绕开 ollama 自带的下载器，手动把模型装进本地 ollama 仓库。

为什么需要这个：
  ollama serve 走 HTTP 代理时会对 registry.ollama.ai 报 "EOF"，
  而 registry.ollama.ai 直连又只有 ~100KB/s（模型主体在 Cloudflare R2 上）。
  用 curl 走代理可以跑到 1.5MB/s，所以这里改成
  "curl 下载 + 手工写入 blob 仓库" 的方式。

用法：
  python3 pull_ollama_model.py <模型名> <标签> [--proxy http://127.0.0.1:7897]
例如：
  python3 pull_ollama_model.py qwen3-vl 2b-instruct

装完用 `ollama list` 应该能看到该模型，且无需联网即可取用。
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request

REGISTRY = "registry.ollama.ai"
MODELS_DIR = os.path.expanduser("~/.ollama/models")
BLOBS_DIR = os.path.join(MODELS_DIR, "blobs")
MANIFEST_DIR = os.path.join(MODELS_DIR, "manifests", REGISTRY, "library")


def log(msg):
    print(msg, flush=True)


def fetch_manifest(name, tag, proxy):
    """取 manifest。manifest 很小，直接用 urllib（走代理）即可。"""
    url = "https://%s/v2/library/%s/manifests/%s" % (REGISTRY, name, tag)
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
    else:
        opener = urllib.request.build_opener()
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
    )
    with opener.open(req, timeout=60) as resp:
        return resp.read()


def blob_path(digest):
    """sha256:abcd... -> ~/.ollama/models/blobs/sha256-abcd..."""
    return os.path.join(BLOBS_DIR, digest.replace(":", "-"))


def blob_url(name, digest):
    return "https://%s/v2/library/%s/blobs/%s" % (REGISTRY, name, digest)


def download_blob(name, digest, size, proxy):
    dest = blob_path(digest)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        log("  已存在，跳过  %s" % digest[:19])
        return True

    # ollama 自己的临时分片文件会干扰，先清掉
    for suffix in ("", ) + tuple("-%d" % i for i in range(64)):
        try:
            os.unlink(dest + "-partial" + suffix)
        except OSError:
            pass

    tmp = dest + ".downloading"
    cmd = [
        "curl", "-sS", "-L", "--retry", "3", "--retry-delay", "2",
        "-C", "-",                     # 断点续传
        "-o", tmp,
        "-w", "%{http_code} %{size_download} %{speed_download}",
        blob_url(name, digest),
    ]
    if proxy:
        cmd[1:1] = ["-x", proxy]

    log("  下载中  %s  (%.1f MB)" % (digest[:19], size / 1024.0 / 1024.0))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stat = (proc.stdout or "").strip().split()
    if not stat or stat[0] not in ("200", "206"):
        log("  失败：%s  %s" % (proc.stdout.strip(), proc.stderr.strip()[:200]))
        return False

    got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if got != size:
        log("  大小不符：期望 %d，实际 %d（保留临时文件以便续传）" % (size, got))
        return False

    os.rename(tmp, dest)
    speed = float(stat[2]) / 1024.0 / 1024.0 if len(stat) > 2 else 0.0
    log("  完成    %.1f MB  (%.2f MB/s)" % (size / 1024.0 / 1024.0, speed))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="模型名，例如 qwen3-vl")
    ap.add_argument("tag", help="标签，例如 2b-instruct")
    ap.add_argument("--proxy", default="http://127.0.0.1:7897")
    args = ap.parse_args()

    os.makedirs(BLOBS_DIR, exist_ok=True)
    manifest_dir = os.path.join(MANIFEST_DIR, args.name)
    os.makedirs(manifest_dir, exist_ok=True)

    log("取 manifest: %s:%s" % (args.name, args.tag))
    raw = fetch_manifest(args.name, args.tag, args.proxy)
    manifest = json.loads(raw.decode("utf-8"))

    targets = [manifest["config"]] + manifest.get("layers", [])
    total = sum(t["size"] for t in targets)
    log("共 %d 个文件，合计 %.1f MB" % (len(targets), total / 1024.0 / 1024.0))

    for t in targets:
        if not download_blob(args.name, t["digest"], t["size"], args.proxy):
            log("中断：有分块未下载完成，重跑本脚本会断点续传。")
            return 1

    manifest_path = os.path.join(manifest_dir, args.tag)
    with open(manifest_path, "wb") as f:
        f.write(raw)
    log("写入 manifest: %s" % manifest_path)
    log("完成。用 `ollama list` 检查。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
