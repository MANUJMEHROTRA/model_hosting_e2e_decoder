# End-to-End Model Hosting Runbook

## Prerequisites
- Python 3.11+, Docker Desktop (with Kubernetes enabled), `kubectl`, `helm`
- 16 GB RAM minimum (model is large; use a GPU machine for Steps 2–4)

---

## Step 1 — Download Data

```bash
pip install datasets pandas
python data/download_data.py
# Creates: data/cnn_dailymail_train.csv (5000 rows)
#          data/cnn_dailymail_validation.csv (500 rows)
#          data/cnn_dailymail_test.csv (500 rows)
```

---

## Step 2 — Train (LoRA + MoRA)

```bash
pip install -r requirements.txt
jupyter lab notebooks/01_training_lora_mora.ipynb
# Outputs: models/lora_merged/   (HuggingFace format)
#          models/lora_adapter.pth
#          models/mora_adapter.pth
#          models/training_results.json
```

---

## Step 3 — Quantization

```bash
jupyter lab notebooks/02_quantization.ipynb
# Outputs: models/quantized/onnx/
#          models/quantized/onnx_int8/
#          models/quantized/quantization_comparison.csv
```

---

## Step 4 — vLLM / SGLang Inference

```bash
# Read the notebook — run cells on a CUDA machine
jupyter lab notebooks/03_vllm_sglang_inference.ipynb
```

---

## Step 5 — Docker (local)

```bash
# Update docker-compose.yml volume path to your absolute models/ path
# Build and start all services
docker compose -f docker/docker-compose.yml up --build

# Test
curl -X POST http://localhost/summarise \
  -H "Content-Type: application/json" \
  -d '{"article": "NASA found water on the moon today..."}'

# View API docs
open http://localhost/docs

# View metrics
curl http://localhost/metrics | grep summarise
```

---

## Step 6 — Kubernetes (Docker Desktop)

```bash
# 1. Enable Kubernetes in Docker Desktop → Settings → Kubernetes → Enable

# 2. Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.2/deploy/static/provider/cloud/deploy.yaml

# 3. Build image so Kubernetes can find it locally
docker build -f docker/Dockerfile -t summarisation-api:latest .

# 4. Update hostPath in 03-deployment.yaml to your absolute models/ path

# 5. Apply all manifests in order
kubectl apply -f k8s/manifests/01-namespace.yaml
kubectl apply -f k8s/manifests/02-configmap.yaml
kubectl apply -f k8s/manifests/03-deployment.yaml
kubectl apply -f k8s/manifests/04-service.yaml
kubectl apply -f k8s/manifests/05-ingress.yaml
kubectl apply -f k8s/manifests/06-hpa.yaml
kubectl apply -f k8s/manifests/07-monitoring.yaml

# 6. Watch pods come up
kubectl get pods -n summarisation -w

# 7. Check pod logs
kubectl logs -n summarisation -l app=summarisation-api --tail=50

# 8. Port-forward to test directly (bypass Ingress)
kubectl port-forward -n summarisation svc/summarisation-service 8080:8080

# 9. View HPA status
kubectl get hpa -n summarisation

# 10. Load test to trigger HPA
for i in $(seq 1 100); do
  curl -s -X POST http://localhost:8080/summarise \
    -H "Content-Type: application/json" \
    -d '{"article": "Test article for load testing..."}' &
done
wait
kubectl get hpa -n summarisation   # watch replicas increase
```

---

## Step 7 — Prometheus + Grafana

```bash
# Prometheus UI
open http://localhost:30090

# Grafana (admin/admin)
open http://localhost:30030

# Key PromQL queries to understand:

# 1. Request rate (requests per second)
sum(rate(summarise_requests_total[1m])) by (endpoint)

# 2. p95 latency — THE most important SLA metric
histogram_quantile(0.95,
  sum(rate(summarise_request_latency_seconds_bucket[5m])) by (le, endpoint)
)

# 3. p90 latency
histogram_quantile(0.90,
  sum(rate(summarise_request_latency_seconds_bucket[5m])) by (le, endpoint)
)

# 4. Error rate
100 * sum(rate(summarise_requests_total{status="error"}[5m]))
    / sum(rate(summarise_requests_total[5m]))

# 5. Apdex score (user satisfaction proxy)
# Satisfied: < 1s, Tolerating: 1-4s, Frustrated: > 4s
(sum(rate(summarise_request_latency_seconds_bucket{le="1"}[5m]))
 + sum(rate(summarise_request_latency_seconds_bucket{le="4"}[5m])) / 2)
/ sum(rate(summarise_request_latency_seconds_count[5m]))
```

---

## Kubernetes Cheat Sheet

```bash
# Resource overview
kubectl get all -n summarisation

# Describe a pod (events, resource usage)
kubectl describe pod -n summarisation <pod-name>

# Shell into a running pod
kubectl exec -it -n summarisation <pod-name> -- /bin/bash

# Watch HPA in real time
watch kubectl get hpa -n summarisation

# Delete everything and start over
kubectl delete namespace summarisation
```

---

## Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
Internet / curl     │           Kubernetes Cluster            │
        │           │                                         │
        ▼           │  ┌──────────────┐                      │
   [NodePort :30080]│  │    Ingress   │  ← nginx-ingress-    │
        │           │  │  Controller  │    controller Pod     │
        ▼           │  └──────┬───────┘                      │
   ┌────────────┐   │         │  routes /summarise            │
   │   NGINX    │   │         ▼                               │
   │  Ingress   │   │  ┌──────────────┐  ClusterIP Service   │
   └─────┬──────┘   │  │  summarise   │◄─────────────────    │
         │          │  │  -service    │                       │
         ▼          │  └──────┬───────┘                      │
   ┌─────────────┐  │         │  load balances               │
   │  Pod 1  API │  │   ┌─────┴──────┬──────────┐           │
   │  Pod 2  API │◄─┼───┤   Pod 1    │  Pod 2   │  Pod 3   │
   │  Pod 3  API │  │   └────────────┴──────────┘           │
   └─────────────┘  │         │                               │
                    │         │ /metrics                      │
                    │         ▼                               │
                    │   ┌───────────┐    ┌─────────┐         │
                    │   │Prometheus │───►│ Grafana │         │
                    │   └───────────┘    └─────────┘         │
                    │   :30090           :30030               │
                    └─────────────────────────────────────────┘
```
