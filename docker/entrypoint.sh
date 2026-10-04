#!/usr/bin/env bash
# Container start: check the GPU, make sure the model is in place, then serve.
#   (no argument)  preflight -> prepare model -> footless serve
#   check          preflight only
#   <anything>     run it as a command (e.g. bash)
set -euo pipefail
cd /app

case "${1:-serve}" in
  check)
    exec python3 /app/docker/preflight.py --check-only
    ;;
  serve)
    ;;
  *)
    exec "$@"
    ;;
esac

# A failed check or download would otherwise be retried by Docker's restart
# policy every few seconds: wait a minute first, so the log stays readable.
stop() {
  echo "[start] not starting; retrying in 60 s (stop with: docker compose down)" >&2
  sleep 60
  exit 1
}
python3 /app/docker/preflight.py || stop
python3 /app/docker/prepare_model.py || stop

PORT="${PORT:-8080}"
args=(serve --model ./models/OrcaSAQ-2-27B/ --host 0.0.0.0 --port "$PORT")
case "${MTP:-1}" in
  1|true|yes|on) args+=(--mtp) ;;
esac
if [[ -n "${API_KEY:-}" ]]; then
  args+=(--api-key "$API_KEY")
fi

shown="python3 -m footless ${args[*]}"
if [[ -n "${API_KEY:-}" ]]; then shown="${shown//"$API_KEY"/***}"; fi
echo "[start] $shown"
echo "[start] loading the weights (seconds to a few minutes, depending on the disk); the API answers on port $PORT once 'serving' is logged"
exec python3 -m footless "${args[@]}"
