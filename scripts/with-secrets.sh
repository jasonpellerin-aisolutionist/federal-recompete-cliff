#!/usr/bin/env bash
# Run a command with the Fed-Spend and SAM.gov keys pulled from Infisical into the environment.
# Values live only in this process; nothing is written to disk.
#
# Settings come from the environment, or from a gitignored .env at the repo root:
#   INFISICAL_DOMAIN, INFISICAL_PROJECT_ID          where the secrets live
#   INFISICAL_UA_CLIENT_ID, INFISICAL_UA_CLIENT_SECRET  machine identity (universal auth)
#   FEDSPEND_SECRET_NAME, SAM_SECRET_NAME           names of the two secrets in the project
#   FEDSPEND_API_BASE                               Fed-Spend API base URL (passed through)
#   INFISICAL_ENV_FILE                              optional extra file defining any of the above
#
# Usage: scripts/with-secrets.sh uv run python -m recompete.fedspend
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && . "$ROOT/.env"
# shellcheck disable=SC1090
[[ -n "${INFISICAL_ENV_FILE:-}" ]] && . "$INFISICAL_ENV_FILE"
set +a

: "${INFISICAL_DOMAIN:?set INFISICAL_DOMAIN}"
: "${INFISICAL_PROJECT_ID:?set INFISICAL_PROJECT_ID}"
: "${INFISICAL_UA_CLIENT_ID:?set INFISICAL_UA_CLIENT_ID}"
: "${INFISICAL_UA_CLIENT_SECRET:?set INFISICAL_UA_CLIENT_SECRET}"
: "${FEDSPEND_SECRET_NAME:?set FEDSPEND_SECRET_NAME}"
: "${SAM_SECRET_NAME:?set SAM_SECRET_NAME}"

TOKEN="$(infisical login --method=universal-auth \
  --client-id="$INFISICAL_UA_CLIENT_ID" --client-secret="$INFISICAL_UA_CLIENT_SECRET" \
  --domain="$INFISICAL_DOMAIN" --silent --plain)"

secret() {
  curl -sf -H "Authorization: Bearer $TOKEN" \
    "$INFISICAL_DOMAIN/api/v3/secrets/raw/$1?workspaceId=$INFISICAL_PROJECT_ID&environment=prod" |
    python3 -c "import sys, json; print(json.load(sys.stdin)['secret']['secretValue'])"
}

FEDSPEND_API_KEY="$(secret "$FEDSPEND_SECRET_NAME")"
SAM_API_KEY="$(secret "$SAM_SECRET_NAME")"
export FEDSPEND_API_KEY SAM_API_KEY
unset TOKEN INFISICAL_UA_CLIENT_SECRET

exec "$@"
