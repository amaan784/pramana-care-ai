#!/usr/bin/env bash
# One-shot deploy: validate bundle, deploy, run all 3 jobs in order, restart app.
set -euo pipefail

TARGET="${1:-prod}"

echo "→ Validate bundle"
databricks bundle validate -t "$TARGET"

echo "→ Deploy bundle"
databricks bundle deploy -t "$TARGET"

echo "→ Run ingest job"
databricks bundle run pramana_ingest -t "$TARGET"

echo "→ Run build_index job"
databricks bundle run pramana_build_index -t "$TARGET"

echo "→ Run deploy_agent job (logs + registers + deploys + evals)"
databricks bundle run pramana_deploy_agent -t "$TARGET"

echo "→ Restart app"
"$(dirname "$0")/restart_app.sh" "$TARGET"

echo "Done."
