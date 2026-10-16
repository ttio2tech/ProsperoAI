#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin a pending media preset from the Hugging Face API and add it to model-tools/presets.json.

  tools/pin-pending-preset.py model-tools/pending-presets/qwen-image-2.1-q4-k.json [--dry-run]

A pending preset names each file's repository and file name but not its revision, size or
SHA-256. This tool resolves each repository's current commit, finds the file in that commit's
tree (by exact path, otherwise by a unique base name), and records the commit, the LFS size and
the LFS SHA-256. The pinned preset replaces any preset with the same ID in presets.json, and
tools/generate-model-presets.py then regenerates include/model_presets.hpp.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = "https://huggingface.co/api/models"


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as response:
        return json.load(response)


def commit_of(repo, cache):
    if repo not in cache:
        sha = get_json(f"{API}/{repo}/revision/main").get("sha", "")
        if len(sha) != 40:
            sys.exit(f"pin: no commit for {repo}")
        tree = get_json(f"{API}/{repo}/tree/{sha}?recursive=1")
        cache[repo] = (sha, [entry for entry in tree if entry.get("type") == "file"])
    return cache[repo]


def find(repo, files, source):
    exact = [entry for entry in files if entry["path"] == source]
    if exact:
        return exact[0]
    base = source.rsplit("/", 1)[-1]
    named = [entry for entry in files if entry["path"].rsplit("/", 1)[-1] == base]
    if len(named) != 1:
        sys.exit(f"pin: {repo} has {len(named)} files named {base}: "
                 f"{[entry['path'] for entry in named]}")
    return named[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pending", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true", help="print the pinned preset only")
    args = parser.parse_args()
    preset = json.loads(args.pending.read_text())
    cache = {}
    for file in preset["files"]:
        repo = file.get("repository", preset["repository"])
        sha, files = commit_of(repo, cache)
        entry = find(repo, files, file["source"])
        lfs = entry.get("lfs") or {}
        if len(lfs.get("oid", "")) != 64:
            sys.exit(f"pin: {repo}/{entry['path']} is not an LFS file with a SHA-256")
        file["source"] = entry["path"]
        file["size"] = lfs.get("size", entry.get("size"))
        file["sha256"] = lfs["oid"]
        if repo == preset["repository"]:
            file.pop("repository", None)
        else:
            file["repository"] = repo
            file["revision"] = sha
        print(f"pin: {repo}@{sha[:12]} {entry['path']} {file['size']} bytes", file=sys.stderr)
    preset["revision"] = commit_of(preset["repository"], cache)[0]
    ordered = {key: preset[key] for key in ["id", "name", "kind", "repository", "revision", "files"]}
    if args.dry_run:
        print(json.dumps(ordered, indent=2))
        return
    path = ROOT / "model-tools/presets.json"
    text = path.read_text()
    if any(p["id"] == preset["id"] for p in json.loads(text)):
        sys.exit(f"pin: {preset['id']} is already in {path.relative_to(ROOT)}; remove it first")
    # Append as text: the existing entries keep their hand-made layout.
    block = "\n".join("  " + line for line in json.dumps(ordered, indent=2).splitlines())
    body = text.rstrip()
    if not body.endswith("]"):
        sys.exit(f"pin: {path.relative_to(ROOT)} does not end with ']'")
    path.write_text(body[:-1].rstrip() + ",\n" + block + "\n]\n")
    json.loads(path.read_text())
    subprocess.run([sys.executable, str(ROOT / "tools/generate-model-presets.py")], check=True)
    print(f"pin: {preset['id']} added to {path.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
