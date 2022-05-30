"""Utility functions."""

import subprocess


def get_git_commit():
    """Return short version of current Git commit hash."""

    short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return str(short_hash, "utf-8").strip()