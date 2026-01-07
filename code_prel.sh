RESOURCE_GROUP="rg-mlops-bank-churn" # changer votre ressource
SUBSCRIPTION_ID=$(az account show --query id -o tsv | tr -d '\r')

# 1. Créer le Service Principal et capturer la sortie
SP_JSON=$(az ad sp create-for-rbac \
  --name "github-actions-$(date +%s)" \
  --role contributor \
  --scopes "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}" \
  --output json)

# 2. Extraire et formater uniquement les 4 champs requis pour GitHub Actions
echo $SP_JSON | jq -c '{clientId: .appId, clientSecret: .password, subscriptionId: "'"$SUBSCRIPTION_ID"'", tenantId: .tenant}'