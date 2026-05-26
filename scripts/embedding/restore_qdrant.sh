#!/usr/bin/env bash
# Restore the `bger` collection from the Hugging Face-hosted snapshot.
#
# Steps:
#   1. Download the snapshot file from the HF dataset repo.
#   2. Verify SHA256 against the manifest in the same repo.
#   3. POST the file to a running Qdrant via /snapshots/recover.
#
# Prerequisites:
#   - hf-cli installed (pip install huggingface_hub[cli])
#   - A running Qdrant instance reachable at $QDRANT_URL (default
#     http://127.0.0.1:6333). For reproducing the thesis runs use the
#     image and config from the kg-rag-legal repository, the collection
#     parameters (size=1024, distance=Cosine) are baked into the snapshot.
#   - curl and sha256sum on PATH.
#
# Usage:
#   ./restore_qdrant.sh [date-tag]
#   e.g. ./restore_qdrant.sh 2026-05-26
#
# If no date tag is passed, the latest file from the manifest is restored.

set -euo pipefail

REPO_ID="${REPO_ID:-albertstudy/graphrag-vs-rag-bger-snapshot}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
COLLECTION="${COLLECTION:-bger}"
WORKDIR="${WORKDIR:-./snapshots}"

mkdir -p "$WORKDIR"

# 1. Download manifest and pick the file to restore.
hf download "$REPO_ID" SHA256SUMS --repo-type dataset --local-dir "$WORKDIR" >/dev/null
DATE_TAG="${1:-}"
if [ -z "$DATE_TAG" ]; then
    # Use the latest entry from the manifest (last line).
    REMOTE_FILE=$(tail -n1 "$WORKDIR/SHA256SUMS" | awk '{print $2}')
else
    REMOTE_FILE="bger-${DATE_TAG}.snapshot"
fi
EXPECTED_SHA=$(grep " ${REMOTE_FILE}$" "$WORKDIR/SHA256SUMS" | awk '{print $1}')
test -n "$EXPECTED_SHA" || { echo "No SHA256 entry for $REMOTE_FILE in manifest" >&2; exit 1; }

echo "Restoring $REMOTE_FILE (expected sha256 $EXPECTED_SHA)"

# 2. Download snapshot.
hf download "$REPO_ID" "$REMOTE_FILE" --repo-type dataset --local-dir "$WORKDIR" >/dev/null
LOCAL="$WORKDIR/$REMOTE_FILE"
ACTUAL_SHA=$(sha256sum "$LOCAL" | awk '{print $1}')
test "$ACTUAL_SHA" = "$EXPECTED_SHA" \
    || { echo "SHA256 mismatch! expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2; exit 1; }
echo "  download OK, checksum verified"

# 3. Restore into Qdrant.
# Qdrant has two restore paths:
#   - /collections/{coll}/snapshots/upload (multipart-upload, recommended).
#   - /collections/{coll}/snapshots/recover (server-side path or URL).
# The upload path works without giving Qdrant filesystem access.
echo "  POSTing snapshot to ${QDRANT_URL}/collections/${COLLECTION}/snapshots/upload …"
curl -sS -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "snapshot=@${LOCAL}" \
    "${QDRANT_URL}/collections/${COLLECTION}/snapshots/upload?priority=snapshot" \
    | tee "$WORKDIR/restore_response.json"
echo
echo "Restore submitted. Verify via:"
echo "  curl ${QDRANT_URL}/collections/${COLLECTION}"
