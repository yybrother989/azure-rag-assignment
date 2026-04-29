#!/usr/bin/env bash
# =====================================================================
# One-shot provisioning for Azure Observable RAG.
#
# Usage: bash infra/deploy.sh <resource-group> <location>
#   e.g. bash infra/deploy.sh kb-rag-rg eastus2
#
# Creates the resource group (if missing), deploys main.bicep, then
# writes a populated .env at the repo root by combining the Bicep
# outputs with `az` queries for the secret keys.
# =====================================================================

set -euo pipefail

RG="${1:-}"
LOC="${2:-eastus2}"
if [[ -z "$RG" ]]; then
  echo "Usage: $0 <resource-group> [location]" >&2
  exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BICEP_FILE="$REPO_ROOT/infra/main.bicep"
ENV_FILE="$REPO_ROOT/.env"

# Canonical tag set — kept in sync with the `tags` parameter default in main.bicep.
# Override per-key via env, e.g.: TAG_owner="alice@example.com" bash deploy.sh ...
TAGS=(
  "project=${TAG_project:-azure-rag-assignment}"
  "environment=${TAG_environment:-demo}"
  "workload=${TAG_workload:-observable-rag}"
  "managedBy=${TAG_managedBy:-bicep}"
  "costCenter=${TAG_costCenter:-take-home}"
)

echo "==> Ensuring resource group '$RG' in '$LOC' with tags: ${TAGS[*]}"
az group create -n "$RG" -l "$LOC" --tags "${TAGS[@]}" --output none

echo "==> Deploying main.bicep (this takes ~5-8 minutes)"
DEPLOY_NAME="kb-rag-$(date +%Y%m%d%H%M%S)"

# Build a JSON tags object for Bicep so the resources inherit any TAG_* overrides.
TAGS_JSON=$(printf '%s\n' "${TAGS[@]}" | python3 -c "
import sys, json
out = {}
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    k, _, v = line.partition('=')
    out[k] = v
print(json.dumps(out))
")

az deployment group create \
  --resource-group "$RG" \
  --name "$DEPLOY_NAME" \
  --template-file "$BICEP_FILE" \
  --parameters location="$LOC" tags="$TAGS_JSON" \
  --output none

echo "==> Reading deployment outputs"
OUTPUTS=$(az deployment group show -g "$RG" -n "$DEPLOY_NAME" --query properties.outputs -o json)

# Bicep lowercases output identifiers up to the first underscore (with a quirk
# for all-caps prefixes, e.g. AZURE_FOO -> azurE_FOO), so look up case-insensitively.
get() { echo "$OUTPUTS" | python3 -c "
import sys, json
outs = json.load(sys.stdin)
target = '$1'.upper()
for k, v in outs.items():
    if k.upper() == target:
        print(v['value']); break
else:
    raise KeyError('$1')
"; }

STORAGE_ACCOUNT=$(get AZURE_STORAGE_ACCOUNT)
STORAGE_CONTAINER=$(get AZURE_STORAGE_CONTAINER)
SEARCH_ENDPOINT=$(get AZURE_SEARCH_ENDPOINT)
SEARCH_INDEX=$(get AZURE_SEARCH_INDEX)
SEMANTIC_CONFIG=$(get AZURE_SEARCH_SEMANTIC_CONFIG)
OPENAI_ENDPOINT=$(get AZURE_OPENAI_ENDPOINT)
EMBED_DEPLOY=$(get AZURE_OPENAI_EMBED_DEPLOYMENT)
CHAT_DEPLOY=$(get AZURE_OPENAI_CHAT_DEPLOYMENT)
FOUNDRY_PROJECT_ENDPOINT=$(get AZURE_AI_FOUNDRY_PROJECT_ENDPOINT)
FOUNDRY_PROJECT_NAME=$(get AZURE_AI_FOUNDRY_PROJECT_NAME)
DI_ENDPOINT=$(get AZURE_DOC_INTELLIGENCE_ENDPOINT)
SEARCH_NAME=$(get SEARCH_RESOURCE_NAME)
AI_SERVICES_NAME=$(get AI_SERVICES_RESOURCE_NAME)
DI_NAME=$(get DOC_INTEL_RESOURCE_NAME)
# Phase 2 — Content Safety
CS_ENDPOINT=$(get AZURE_CONTENT_SAFETY_ENDPOINT)
CS_NAME=$(get CONTENT_SAFETY_RESOURCE_NAME)
# Phase 3 — Application Insights
APP_INSIGHTS_CONN=$(get APPLICATIONINSIGHTS_CONNECTION_STRING)
# Phase 5 — Function App
FUNC_APP_NAME=$(get FUNCTION_APP_NAME)
FUNC_APP_PRINCIPAL_ID=$(get FUNCTION_APP_PRINCIPAL_ID)

echo "==> Fetching admin keys (kept for fallback; Foundry path uses AAD)"
SEARCH_KEY=$(az search admin-key show -g "$RG" --service-name "$SEARCH_NAME" --query primaryKey -o tsv)
OPENAI_KEY=$(az cognitiveservices account keys list -g "$RG" -n "$AI_SERVICES_NAME" --query key1 -o tsv)
DI_KEY=$(az cognitiveservices account keys list -g "$RG" -n "$DI_NAME" --query key1 -o tsv)
CS_KEY=$(az cognitiveservices account keys list -g "$RG" -n "$CS_NAME" --query key1 -o tsv)

echo "==> Granting current user 'Storage Blob Data Contributor' on the storage account"
PRINCIPAL_ID=$(az ad signed-in-user show --query id -o tsv)
STORAGE_SCOPE=$(az storage account show -g "$RG" -n "$STORAGE_ACCOUNT" --query id -o tsv)
az role assignment create \
  --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type User \
  --role "Storage Blob Data Contributor" \
  --scope "$STORAGE_SCOPE" \
  --output none 2>/dev/null || echo "  (role assignment already exists or insufficient privilege)"

echo "==> Granting current user 'Azure AI Developer' on the AI Services account (for Foundry project AAD access)"
AI_SERVICES_SCOPE=$(az cognitiveservices account show -g "$RG" -n "$AI_SERVICES_NAME" --query id -o tsv)
az role assignment create \
  --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type User \
  --role "Azure AI Developer" \
  --scope "$AI_SERVICES_SCOPE" \
  --output none 2>/dev/null || echo "  (role assignment already exists or insufficient privilege)"
# Cognitive Services User role grants data-plane access (chat/embeddings calls) via AAD
az role assignment create \
  --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type User \
  --role "Cognitive Services User" \
  --scope "$AI_SERVICES_SCOPE" \
  --output none 2>/dev/null || echo "  (Cognitive Services User role already exists)"

echo "==> Granting current user 'Cognitive Services User' on Content Safety + Doc Intel"
for resource_name in "$CS_NAME" "$DI_NAME"; do
  scope=$(az cognitiveservices account show -g "$RG" -n "$resource_name" --query id -o tsv)
  az role assignment create \
    --assignee-object-id "$PRINCIPAL_ID" \
    --assignee-principal-type User \
    --role "Cognitive Services User" \
    --scope "$scope" \
    --output none 2>/dev/null || echo "  ($resource_name role already exists)"
done

echo "==> Granting Function App MSI access to Storage / Search / DI / AI Services"
# Function App needs to: read kb-docs, upload kb-figures, query/upload to Search,
# and call DI + AOAI.
for role_scope in \
  "Storage Blob Data Contributor|$STORAGE_SCOPE" \
  "Search Index Data Contributor|$(az search service show -g $RG -n $SEARCH_NAME --query id -o tsv)" \
  "Cognitive Services User|$AI_SERVICES_SCOPE" \
  "Cognitive Services User|$(az cognitiveservices account show -g $RG -n $DI_NAME --query id -o tsv)"; do
  role="${role_scope%%|*}"
  scope="${role_scope#*|}"
  az role assignment create \
    --assignee-object-id "$FUNC_APP_PRINCIPAL_ID" \
    --assignee-principal-type ServicePrincipal \
    --role "$role" \
    --scope "$scope" \
    --output none 2>/dev/null || echo "  (Function MSI: $role already exists or skipped)"
done

echo "==> Configuring Function App settings required by the ingest runtime"
az functionapp config appsettings set \
  -g "$RG" -n "$FUNC_APP_NAME" \
  --settings \
    "SCM_DO_BUILD_DURING_DEPLOYMENT=true" \
    "ENABLE_ORYX_BUILD=true" \
    "AZURE_SEARCH_API_KEY=$SEARCH_KEY" \
    "AZURE_OPENAI_API_KEY=$OPENAI_KEY" \
    "AZURE_AI_FOUNDRY_PROJECT_ENDPOINT=$FOUNDRY_PROJECT_ENDPOINT" \
    "AZURE_AI_FOUNDRY_PROJECT_NAME=$FOUNDRY_PROJECT_NAME" \
    "AZURE_DOC_INTELLIGENCE_KEY=$DI_KEY" \
    "AZURE_FIGURES_CONTAINER=kb-figures" \
    "ENABLE_FIGURE_CAPTIONS=0" \
  --output none

echo "==> Deploying Function code (zip from infra/functions/)"
FUNC_ZIP="/tmp/kb-funcs-$$.zip"
# Copy src/ into the function package so the blob trigger can `from src.ingest import ingest_single`.
TMP_FUNC_DIR=$(mktemp -d)
cp -R "$REPO_ROOT/infra/functions/." "$TMP_FUNC_DIR/"
cp -R "$REPO_ROOT/src" "$TMP_FUNC_DIR/src"
( cd "$TMP_FUNC_DIR" && zip -qr "$FUNC_ZIP" . )
az functionapp deployment source config-zip \
  -g "$RG" -n "$FUNC_APP_NAME" --src "$FUNC_ZIP" --build-remote true \
  --output none 2>&1 | tail -3 || echo "  (function deploy may need a manual retry)"
rm -rf "$TMP_FUNC_DIR" "$FUNC_ZIP"

echo "==> Writing $ENV_FILE"
cat > "$ENV_FILE" <<EOF
AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT
AZURE_STORAGE_CONTAINER=$STORAGE_CONTAINER

AZURE_SEARCH_ENDPOINT=$SEARCH_ENDPOINT
AZURE_SEARCH_INDEX=$SEARCH_INDEX
AZURE_SEARCH_API_KEY=$SEARCH_KEY
AZURE_SEARCH_SEMANTIC_CONFIG=$SEMANTIC_CONFIG

# Primary model access goes through Azure AI Foundry project (AAD-authed).
AZURE_AI_FOUNDRY_PROJECT_ENDPOINT=$FOUNDRY_PROJECT_ENDPOINT
AZURE_AI_FOUNDRY_PROJECT_NAME=$FOUNDRY_PROJECT_NAME

# Fallback path — direct AOAI-style access against the same AI Services account.
AZURE_OPENAI_ENDPOINT=$OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY=$OPENAI_KEY
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_EMBED_DEPLOYMENT=$EMBED_DEPLOY
AZURE_OPENAI_CHAT_DEPLOYMENT=$CHAT_DEPLOY

AZURE_DOC_INTELLIGENCE_ENDPOINT=$DI_ENDPOINT
AZURE_DOC_INTELLIGENCE_KEY=$DI_KEY

# Phase 2 — Content Safety
AZURE_CONTENT_SAFETY_ENDPOINT=$CS_ENDPOINT
AZURE_CONTENT_SAFETY_KEY=$CS_KEY

# Phase 3 — observability
APPLICATIONINSIGHTS_CONNECTION_STRING=$APP_INSIGHTS_CONN

CHUNK_MAX_TOKENS=512
CHUNK_OVERLAP_TOKENS=50
RETRIEVAL_TOP_K=5
EVIDENCE_TOP_N=4
LOG_TRACE_JSONL=traces.jsonl
EOF

chmod 600 "$ENV_FILE"
echo "==> Done. .env written. Next: drop docs into ./data and run 'python -m src.ingest'"
