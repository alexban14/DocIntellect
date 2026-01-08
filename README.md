# DocIntellect

## A system for intelligent document understanding and retrieval

## Build and deploy docker images

### Client SPA
```bash
docker buildx build --pull \
  --platform linux/amd64 \
  --progress=plain \
  --file ./client/Dockerfile \
  --tag ghcr.io/alexban14/plumbrain/client:latest \
  --push --provenance=false ./client
```

### LLM Interaction Service
```bash
docker buildx build --pull \
  --platform linux/amd64 \
  --progress=plain \
  --file ./llm_interaction_service/Dockerfile \
  --tag ghcr.io/alexban14/plumbrain/llm_interaction_service:latest \
  --push --provenance=false ./llm_interaction_service
```
