# COLLECTEUR PV Output - ReadMe

## PURPOSE
Aller récupérer sur l'API de PVOutput.org les données d'une station cible :
- données production "live" (cad pas de temps de 5min à 10min sans la météo) ==> Detail_PROD
- données d'insolation (cad pas de temps de 5min à 10min sans la météo. Pas complètement clair si correspond exactement extraterestrial radiation) ==> Detail_INSOL
- données production quotidienne (somme sur la journée, dispo aussi en mensuel etc. Ajoute un champ (catégoriel) sur la couverture nuageuse) ==> Aggreg_PROD

## PRINCIPE
Traitement successif des 3 types de requêtes (avec en bonus les carac de l'installation)
- Formulation et test de la requêtes
- Définition des dates:
- Déterminer la liste des dates maximale
    - A partir des logs, déterminer les dates pour lesquelles on a déjà les données
    - Etablir la nouvelle liste de dates pour lesquelles on doit requêter (nouvelles dates ou erreurs)
- Boucle de requêtes
    - Charger le DF sur disque
    - Envois requête
    - Append des données reçues dans le DF
- Exporter les résultat prk et csv)
    - Logs
    - Data

## CHANGELOGS : 
- v2.2 Ajout de la gestion des dates déjà téléchargés

## TODO :
- le rappel des date dl ne fonctionne pas comme il faut
- la borne haute des dates à appelée n'est pas call (exclusif au lieu d'inclusif)
- appliquer la refactoriser en fonctions au autres url
    - <<Attention ! pr les données aggrégées il va falloir reprendre des choses
- convertir les types à la création des df
- merge les dates et time en un TS ?
- Gestion des limites d'API
    - "Rate limit information can be obtained by setting the HTTP header X-Rate-Limit to 1" https://pvoutput.org/help.html

