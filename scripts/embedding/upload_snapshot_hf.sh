#!/usr/bin/env bash
# Upload the Qdrant `bger` collection snapshot to a Hugging Face dataset
# repository for thesis-grade reproducibility.
#
# Run this once after creating a fresh snapshot. The dataset repository
# `albertstudy/graphrag-vs-rag-bger-snapshot` hosts a single file,
# `bger-YYYY-MM-DD.snapshot`, plus a `SHA256SUMS` manifest.
#
# Prerequisites:
#   - hf-cli installed (pip install huggingface_hub[cli])
#   - HF_TOKEN env var or `hf auth login` already done
#   - The snapshot file present locally
#
# Usage:
#   HF_TOKEN=hf_xxx ./upload_snapshot_hf.sh <path-to-snapshot> [date-tag]
#   e.g.
#   HF_TOKEN=hf_xxx ./upload_snapshot_hf.sh /data/thesis/snapshots/bger-2026-05-26.snapshot 2026-05-26

set -euo pipefail

REPO_ID="${REPO_ID:-albertstudy/graphrag-vs-rag-bger-snapshot}"
SNAPSHOT_PATH="${1:?usage: $0 <snapshot-path> [date-tag]}"
DATE_TAG="${2:-$(date +%Y-%m-%d)}"
REMOTE_FILE="bger-${DATE_TAG}.snapshot"

test -f "$SNAPSHOT_PATH" || { echo "Snapshot not found: $SNAPSHOT_PATH" >&2; exit 1; }

SIZE=$(stat -c %s "$SNAPSHOT_PATH")
SHA=$(sha256sum "$SNAPSHOT_PATH" | awk '{print $1}')

echo "Snapshot file: $SNAPSHOT_PATH"
echo "  size: $((SIZE / 1024 / 1024 / 1024)) GiB ($SIZE bytes)"
echo "  sha256: $SHA"
echo "  remote: ${REPO_ID}:${REMOTE_FILE}"

# 1. Create dataset repo if it doesn't exist (idempotent).
hf repo create "$REPO_ID" --type dataset --exist-ok

# 2. Upload the snapshot file. Resumable, chunked.
# Xet-Backend disabled, large legacy uploads via plain LFS are more
# reliable for files in the 20+ GiB range and do not depend on the
# xet-cache directory being writable for the running user.
HF_HUB_DISABLE_XET=1 hf upload "$REPO_ID" "$SNAPSHOT_PATH" "$REMOTE_FILE" --repo-type dataset

# 3. Write/update SHA256SUMS manifest in the repo root.
TMP=$(mktemp)
echo "$SHA  $REMOTE_FILE" > "$TMP"
hf upload "$REPO_ID" "$TMP" "SHA256SUMS" --repo-type dataset
rm -f "$TMP"

cat <<EOF

Upload complete. The snapshot is now downloadable via:

  hf download $REPO_ID $REMOTE_FILE \\
      --repo-type dataset --local-dir ./snapshots

Restore into a fresh Qdrant instance with scripts/embedding/restore_qdrant.sh.
EOF
