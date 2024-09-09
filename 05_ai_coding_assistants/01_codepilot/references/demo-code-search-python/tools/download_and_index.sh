#!/usr/bin/env bash

set -e

# Ensure current path is project root
cd "$(dirname "$0")/../"

git clone https://github.com/qdrant/qdrant.git /tmp/repo

REPO_PATH=/tmp/repo bash -x tools/index_repo.sh /tmp/repo

rm -rf /tmp/repo

