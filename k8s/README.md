# ☸️ Manifestes Kubernetes / OpenShift

Déploiement de la **plateforme** GraphRAG (API FastAPI + interface Streamlit) sur un
cluster. Les scripts de démo (`demo_*.py`) ne sont pas concernés : ils ne sont pas dans
l'image.

La base de données **n'est pas déployée ici** — Neo4j est externe (Aura), et c'est elle qui
calcule les embeddings en appelant OpenAI. Ces manifestes ne décrivent qu'une application
sans état.

| Fichier | Rôle | Dans `kustomization.yaml` |
|---|---|---|
| `namespace.yaml` | Le namespace `graphrag` | ✅ |
| `configmap.yaml` | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_DATABASE` | ✅ |
| `secret.example.yaml` | Modèle pour `OPENAI_API_KEY` / `NEO4J_PASSWORD` | ❌ à créer hors dépôt |
| `deployment.yaml` | Le pod, ses sondes, ses limites | ✅ |
| `service.yaml` | 8501 (interface) + 8000 (diagnostic seul) | ✅ |
| `ingress.yaml` | Publication de l'interface, affinité de session | ✅ |
| `networkpolicy.yaml` | Cloisonnement réseau | ❌ opt-in, à adapter |
| `openshift/route.yaml` | Remplace l'Ingress sur OpenShift | ❌ |

## Déployer

```bash
# 1. Le secret, hors du dépôt
kubectl create namespace graphrag
kubectl -n graphrag create secret generic graphrag-secrets \
  --from-literal=OPENAI_API_KEY='sk-...' \
  --from-literal=NEO4J_PASSWORD='...'

# 2. Renseigner NEO4J_URI dans configmap.yaml, l'hôte dans ingress.yaml,
#    et épingler une version d'image dans kustomization.yaml

# 3. Appliquer
kubectl apply -k k8s/
kubectl -n graphrag rollout status deploy/graphrag
```

Sur **OpenShift**, tout est identique sauf la publication :

```bash
oc apply -f k8s/namespace.yaml -f k8s/configmap.yaml \
         -f k8s/deployment.yaml -f k8s/service.yaml
oc apply -f k8s/openshift/route.yaml
oc -n graphrag get route graphrag -o jsonpath='{.spec.host}{"\n"}'
```

## Vérifier

```bash
# Le vrai état : /health renvoie toujours HTTP 200, l'information est dans le corps
kubectl -n graphrag exec deploy/graphrag -- curl -s localhost:8000/health

kubectl -n graphrag logs deploy/graphrag | head -20   # contrôles de docker-start.sh
kubectl -n graphrag get events --sort-by=.lastTimestamp
kubectl -n graphrag port-forward svc/graphrag 8501:8501
```

## Ce qu'il faut savoir avant de modifier ces fichiers

**Un seul Deployment, deux processus.** `docker-start.sh` lance uvicorn en arrière-plan
puis streamlit. Les scinder en deux workloads exige de modifier le code : `API_BASE_URL`
est codé en dur à `http://localhost:8000` dans `streamlit_rag_simple.py:16`.

**`GET /health` renvoie toujours HTTP 200**, y compris quand Neo4j est injoignable — l'état
est dans le corps JSON (`KnowledgeGraphRagAPI/main.py:1053`). D'où la `readinessProbe` en
`exec` : une sonde `httpGet` sur `/health` ne pourrait jamais échouer.

**`replicas: 1`.** La session Streamlit est à état et portée par un websocket. Au-delà d'un
réplica, l'affinité de session devient obligatoire (annotations présentes dans
`ingress.yaml`, native sur les Routes OpenShift).

**`HOME=/tmp` n'est pas cosmétique.** L'image crée son utilisateur sans `-m` :
`/home/graphrag` n'existe pas. Sous OpenShift, l'UID est en plus aléatoire et absent de
`/etc/passwd`. Sans `HOME` inscriptible, Streamlit ne démarre pas.

**`runAsNonRoot: true` est volontairement absent.** Le Dockerfile déclare un `USER` non
numérique, que le kubelet ne peut pas valider : le pod serait refusé. Ajoutez
`runAsUser: 1001` si votre politique l'exige.

**Le port 8000 n'est pas publié.** `POST /cypher` exécute du Cypher arbitraire, écritures
comprises, sans authentification. Il reste dans le `Service` pour le seul `port-forward`.

**Aucune authentification, nulle part.** Ni l'interface ni l'API n'en ont. Une exposition
publique impose une couche en amont (OAuth2 Proxy, `oauth-proxy` en side-car sur OpenShift,
Entra ID sur Azure).

📖 Le contexte complet — CI/CD, OpenShift, les trois cibles Azure, et ce qui manque encore
pour une vraie production : **[§ DevOps & déploiement orchestré](../README.md#️-devops--déploiement-orchestré)**.
