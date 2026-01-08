RESOURCE_GROUP="rg-mlops-bank-churn-italy"
LOCATION="francecentral"                 # <-- met la région désirée
CONTAINER_APP_NAME="bank-churn"

# Création d'Application Insights
az monitor app-insights component create \
  --app bank-churn-insights \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --application-type web

# Récupération de la connection string
APPINSIGHTS_CONN=$(az monitor app-insights component show \
  --app bank-churn-insights \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv)

echo "Connection String : $APPINSIGHTS_CONN"

# Injection de la variable d'environnement dans Azure Container Apps
#Ton application, lorsqu’elle démarre \n
#lira cette variable d’environnement pour savoir où envoyer ses logs
# et métriques. Sans ça, Application Insights ne pourra pas recevoir les données de ton app.
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars "APPLICATIONINSIGHTS_CONNECTION_STRING=$APPINSIGHTS_CONN"