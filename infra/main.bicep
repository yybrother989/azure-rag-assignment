// =====================================================================
// Azure Observable RAG — one-shot infrastructure
// Provisions: Storage + container, AI Search (Standard), Azure OpenAI
// (with embedding + chat deployments), Document Intelligence.
//
// Outputs everything the Python pipeline needs in .env (see deploy.sh).
// =====================================================================

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Suffix appended to all globally-unique resource names')
param nameSuffix string = uniqueString(resourceGroup().id)

@description('Azure AI Search SKU. Semantic ranker requires standard or higher.')
@allowed([ 'standard', 'standard2', 'standard3' ])
param searchSku string = 'standard'

@description('Embedding model deployment name')
param embedDeploymentName string = 'text-embedding-3-small'

@description('Chat model deployment name (also used as the env var AZURE_OPENAI_CHAT_DEPLOYMENT)')
param chatDeploymentName string = 'gpt-4o'

@description('Tags applied to every resource. deploy.sh applies the same set to the resource group.')
param tags object = {
  project: 'azure-rag-assignment'
  environment: 'demo'
  workload: 'observable-rag'
  managedBy: 'bicep'
  costCenter: 'take-home'
}

@description('Name of the Azure AI Foundry project created inside the AI Services account')
param foundryProjectName string = 'kb-rag-project'

// ---------- Storage ----------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'kbrag${nameSuffix}'
  location: location
  tags: tags
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    // Allow per-container public access so the kb-figures container can serve
    // PNG crops directly to Chainlit's `cl.Image(url=...)`. The kb-docs
    // container stays private (publicAccess: 'None'). Tighten with SAS in prod.
    allowBlobPublicAccess: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource kbContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'kb-docs'
  properties: { publicAccess: 'None' }
}

// Figure crops extracted by Document Intelligence. Public-blob read so
// Chainlit can render `cl.Image(url=...)` directly. Acceptable for this
// demo because figures came from docs we shipped; tighten with SAS in prod.
resource figuresContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'kb-figures'
  properties: { publicAccess: 'Blob' }
}

// ---------- Azure AI Search ----------
resource search 'Microsoft.Search/searchServices@2024-03-01-preview' = {
  name: 'kb-search-${nameSuffix}'
  location: location
  tags: tags
  sku: { name: searchSku }
  properties: {
    replicaCount: 1
    partitionCount: 1
    semanticSearch: 'standard'
    publicNetworkAccess: 'enabled'
  }
}

// ---------- Azure AI Foundry — AI Services account + project ----------
// Single account hosts model deployments AND the Foundry project. The project
// gives us a unified endpoint for AIProjectClient.get_openai_client().
resource aiServices 'Microsoft.CognitiveServices/accounts@2026-03-01' = {
  name: 'kb-aif-${nameSuffix}'
  location: location
  tags: tags
  sku: { name: 'S0' }
  kind: 'AIServices'
  identity: { type: 'SystemAssigned' }
  properties: {
    customSubDomainName: 'kb-aif-${nameSuffix}'
    publicNetworkAccess: 'Enabled'
    allowProjectManagement: true
    disableLocalAuth: false  // Keep API-key auth available as fallback
  }
}

resource embedDeployment 'Microsoft.CognitiveServices/accounts/deployments@2026-03-01' = {
  parent: aiServices
  name: embedDeploymentName
  // text-embedding-3-small in swedencentral only supports GlobalStandard /
  // DataZoneStandard — regional Standard isn't offered. 1000 K TPM available;
  // capacity 50 = 50 K TPM, plenty of headroom.
  sku: { name: 'GlobalStandard', capacity: 50 }
  properties: {
    model: { format: 'OpenAI', name: 'text-embedding-3-small', version: '1' }
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2026-03-01' = {
  parent: aiServices
  name: chatDeploymentName
  // gpt-4o (full) — gpt-4o-mini is deprecation-frozen for new deployments and
  // gpt-4.1-mini only has quota in eastus2 (currently capacity-out). Trial sub
  // gets 50 K TPM regional Standard for gpt-4o in swedencentral.
  sku: { name: 'Standard', capacity: 20 }
  properties: {
    model: { format: 'OpenAI', name: 'gpt-4o', version: '2024-11-20' }
  }
  dependsOn: [ embedDeployment ]
}

resource foundryProject 'Microsoft.CognitiveServices/accounts/projects@2026-03-01' = {
  parent: aiServices
  name: foundryProjectName
  location: location
  tags: tags
  identity: { type: 'SystemAssigned' }
  properties: {
    displayName: 'KB RAG Project'
    description: 'Azure Observable RAG — unified model access via Foundry project'
  }
  dependsOn: [ chatDeployment ]
}

// ---------- Document Intelligence (primary extractor in Phase 1+) ----------
resource docIntel 'Microsoft.CognitiveServices/accounts@2026-03-01' = {
  name: 'kb-di-${nameSuffix}'
  location: location
  tags: tags
  sku: { name: 'S0' }
  kind: 'FormRecognizer'
  properties: {
    customSubDomainName: 'kb-di-${nameSuffix}'
    publicNetworkAccess: 'Enabled'
  }
}

// ---------- Azure AI Content Safety (Phase 2: prompt + answer filter) ----------
resource contentSafety 'Microsoft.CognitiveServices/accounts@2026-03-01' = {
  name: 'kb-cs-${nameSuffix}'
  location: location
  tags: tags
  sku: { name: 'S0' }
  kind: 'ContentSafety'
  properties: {
    customSubDomainName: 'kb-cs-${nameSuffix}'
    publicNetworkAccess: 'Enabled'
  }
}

// ---------- Log Analytics + Application Insights (Phase 3: observability) ----------
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'kb-logs-${nameSuffix}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'kb-ai-${nameSuffix}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
  }
}

// ---------- Function App (Phase 5: blob-trigger auto-ingest) ----------
// Linux Consumption Plan (Y1) — pay-per-execution, ~$0 idle.
resource functionPlan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: 'kb-funcplan-${nameSuffix}'
  location: location
  tags: tags
  sku: { name: 'Y1', tier: 'Dynamic' }
  kind: 'linux'
  properties: { reserved: true }
}

resource functionApp 'Microsoft.Web/sites@2023-12-01' = {
  name: 'kb-funcs-${nameSuffix}'
  location: location
  tags: tags
  kind: 'functionapp,linux'
  identity: { type: 'SystemAssigned' }
  properties: {
    serverFarmId: functionPlan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'Python|3.11'
      appSettings: [
        { name: 'AzureWebJobsStorage', value: 'DefaultEndpointsProtocol=https;AccountName=${storage.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storage.listKeys().keys[0].value}' }
        { name: 'FUNCTIONS_EXTENSION_VERSION', value: '~4' }
        { name: 'FUNCTIONS_WORKER_RUNTIME', value: 'python' }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
        { name: 'SCM_DO_BUILD_DURING_DEPLOYMENT', value: 'true' }
        { name: 'ENABLE_ORYX_BUILD', value: 'true' }
        { name: 'AZURE_STORAGE_ACCOUNT', value: storage.name }
        { name: 'AZURE_STORAGE_CONTAINER', value: kbContainer.name }
        { name: 'AZURE_SEARCH_ENDPOINT', value: 'https://${search.name}.search.windows.net' }
        { name: 'AZURE_SEARCH_INDEX', value: 'kb-chunks' }
        { name: 'AZURE_SEARCH_SEMANTIC_CONFIG', value: 'kb-semantic' }
        { name: 'AZURE_OPENAI_ENDPOINT', value: aiServices.properties.endpoint }
        { name: 'AZURE_AI_FOUNDRY_PROJECT_ENDPOINT', value: 'https://${aiServices.name}.services.ai.azure.com/api/projects/${foundryProject.name}' }
        { name: 'AZURE_AI_FOUNDRY_PROJECT_NAME', value: foundryProject.name }
        { name: 'AZURE_OPENAI_API_VERSION', value: '2024-10-21' }
        { name: 'AZURE_OPENAI_EMBED_DEPLOYMENT', value: embedDeployment.name }
        { name: 'AZURE_OPENAI_CHAT_DEPLOYMENT', value: chatDeployment.name }
        { name: 'AZURE_DOC_INTELLIGENCE_ENDPOINT', value: docIntel.properties.endpoint }
        { name: 'AZURE_FIGURES_CONTAINER', value: figuresContainer.name }
        { name: 'ENABLE_FIGURE_CAPTIONS', value: '0' }
      ]
    }
  }
}

// ---------- Outputs ----------
output AZURE_STORAGE_ACCOUNT string = storage.name
output AZURE_STORAGE_CONTAINER string = kbContainer.name
output AZURE_SEARCH_ENDPOINT string = 'https://${search.name}.search.windows.net'
output AZURE_SEARCH_INDEX string = 'kb-chunks'
output AZURE_SEARCH_SEMANTIC_CONFIG string = 'kb-semantic'
output AZURE_OPENAI_ENDPOINT string = aiServices.properties.endpoint
output AZURE_OPENAI_EMBED_DEPLOYMENT string = embedDeployment.name
output AZURE_OPENAI_CHAT_DEPLOYMENT string = chatDeployment.name
output AZURE_AI_FOUNDRY_PROJECT_ENDPOINT string = 'https://${aiServices.name}.services.ai.azure.com/api/projects/${foundryProject.name}'
output AZURE_AI_FOUNDRY_PROJECT_NAME string = foundryProject.name
output AZURE_DOC_INTELLIGENCE_ENDPOINT string = docIntel.properties.endpoint

// Phase 2 — Content Safety
output AZURE_CONTENT_SAFETY_ENDPOINT string = contentSafety.properties.endpoint

// Phase 3 — observability
output APPLICATIONINSIGHTS_CONNECTION_STRING string = appInsights.properties.ConnectionString
output LOG_ANALYTICS_WORKSPACE_ID string = logAnalytics.id

// Phase 5 — Function App
output FUNCTION_APP_NAME string = functionApp.name
output FUNCTION_APP_HOSTNAME string = functionApp.properties.defaultHostName
output FUNCTION_APP_PRINCIPAL_ID string = functionApp.identity.principalId

// Resource names (deploy.sh uses for `az` queries that need the resource name, not just endpoint)
output SEARCH_RESOURCE_NAME string = search.name
output AI_SERVICES_RESOURCE_NAME string = aiServices.name
output DOC_INTEL_RESOURCE_NAME string = docIntel.name
output CONTENT_SAFETY_RESOURCE_NAME string = contentSafety.name
