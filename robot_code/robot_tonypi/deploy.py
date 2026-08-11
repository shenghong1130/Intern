#!/usr/bin/env python3
"""Deploy competition_tonypi to the robot via paramiko SFTP."""

import os
import sys
import paramiko

HOST = "192.168.31.138"
USER = "pi"
PASS = "pi"
REMOTE_ROOT = "/home/pi/TonyPi/competition_tonypi"
LOCAL_ROOT = os.path.dirname(os.path.abspath(__file__))

SKIP_DIRS = {"__pycache__", ".git", ".vscode"}
SKIP_SUFFIXES = {".pyc"}


def should_upload(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    if any(p in SKIP_DIRS for p in parts):
        return False
    if any(rel_path.endswith(s) for s in SKIP_SUFFIXES):
        return False
    return True


def collect_files(local_root: str) -> list:
    files = []
    for dirpath, dirnames, filenames in os.walk(local_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, local_root)
            if should_upload(rel_path):
                files.append((rel_path, abs_path))
    return files


def collect_dirs(local_root: str) -> list:
    dirs = set()
    for dirpath, dirnames, filenames in os.walk(local_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, local_root)
            if should_upload(rel_path):
                parent = os.path.dirname(rel_path)
                if parent:
                    dirs.add(parent)
    return sorted(dirs)


def main():
    print(f"Connecting to {USER}@{HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=10)
    print("Connected.")

    # Check remote root exists
    stdin, stdout, stderr = ssh.exec_command(f"ls {REMOTE_ROOT}/main.py")
    out = stdout.read().decode().strip()
    if not out:
        print(f"ERROR: {REMOTE_ROOT}/main.py not found on remote.")
        ssh.close()
        return 1
    print(f"Remote directory confirmed: {REMOTE_ROOT}")

    # Collect files and dirs
    files = collect_files(LOCAL_ROOT)
    dirs = collect_dirs(LOCAL_ROOT)
    print(f"Files to sync: {len(files)}, subdirectories: {len(dirs)}")

    # Create all remote subdirectories via ssh (mkdir -p is reliable)
    if dirs:
        remote_dirs = [REMOTE_ROOT + "/" + d.replace("\\", "/") for d in dirs]
        mkdir_cmd = "mkdir -p " + " ".join(f'"{d}"' for d in remote_dirs)
        stdin, stdout, stderr = ssh.exec_command(mkdir_cmd)
        err = stderr.read().decode().strip()
        if err:
            print(f"mkdir warning: {err}")
        print(f"Created directories on remote.")

    # Upload files via SFTP
    sftp = ssh.open_sftp()
    uploaded = 0
    errors = 0
    for rel_path, abs_path in files:
        remote_path = REMOTE_ROOT + "/" + rel_path.replace("\\", "/")
        try:
            sftp.put(abs_path, remote_path)
            uploaded += 1
        except Exception as exc:
            print(f"  ERROR uploading {rel_path}: {exc}")
            errors += 1

    sftp.close()
    ssh.close()
    print(f"Done: {uploaded} uploaded, {errors} errors")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
