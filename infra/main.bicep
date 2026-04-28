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
    allowBlobPublicAccess: false
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

// ---------- Document Intelligence (optional fallback) ----------
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
output SEARCH_RESOURCE_NAME string = search.name
output AI_SERVICES_RESOURCE_NAME string = aiServices.name
output DOC_INTEL_RESOURCE_NAME string = docIntel.name
