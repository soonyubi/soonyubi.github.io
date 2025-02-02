---
title: Loki, Grafana, OpenTelemetry, Kubernetes를 활용한 로그 확인
type: blog
prev: /
next: docs/folder/
---

{{% steps %}}

### 환경준비

This is the first step.

### nest.js / fastapi 앱 준비

This is the second step.

### k8s에 앱 배포

This is the second step.

### nodejs opentelemetry 구축

1. 관련패키지 설치

```
npm install @nestjs/terminus @opentelemetry/sdk-node @opentelemetry/api @opentelemetry/sdk-trace-node @opentelemetry/sdk-metrics @opentelemetry/instrumentation @opentelemetry/instrumentation-http @opentelemetry/instrumentation-express @opentelemetry/exporter-trace-otlp-http @opentelemetry/exporter-metrics-otlp-http
```

### Helm으로 Loki 설치

```values.yaml
loki:
  commonConfig:
    replication_factor: 1
  schemaConfig:
    configs:
      - from: "2024-04-01"
        store: tsdb
        object_store: s3
        schema: v13
        index:
          prefix: loki_index_
          period: 24h
  pattern_ingester:
    enabled: true
  limits_config:
    allow_structured_metadata: true
    volume_enabled: true
    retention_period: 672h # 28 days retention
  compactor:
    retention_enabled: true
    delete_request_store: s3
  ruler:
    enable_api: true

minio:
  enabled: true

deploymentMode: SingleBinary

singleBinary:
  replicas: 1
  resources:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "500m"

# Zero out replica counts of other deployment modes
backend:
  replicas: 0
read:
  replicas: 0
write:
  replicas: 0

ingester:
  replicas: 0
querier:
  replicas: 0
queryFrontend:
  replicas: 0
queryScheduler:
  replicas: 0
distributor:
  replicas: 0
compactor:
  replicas: 0
indexGateway:
  replicas: 0
bloomCompactor:
  replicas: 0
bloomGateway:
  replicas: 0

```

클러스터 내부에서 로그 전송
• Cluster DNS 주소: 클러스터 내부에서 Loki Gateway를 통해 로그를 전송할 수 있는 API URL.
• 이 URL을 사용하여 클러스터 내부 애플리케이션에서 Loki로 로그를 보낼 수 있음.

```
http://loki-gateway.monitoring.svc.cluster.local/loki/api/v1/push
```

로그 전송 테스트

Loki로 로그를 전송하는 테스트 방법:

```
curl -H "Content-Type: application/json" -XPOST -s "http://127.0.0.1:3100/loki/api/v1/push"  \
--data-raw "{\"streams\": [{\"stream\": {\"job\": \"test\"}, \"values\": [[\"$(date +%s)000000000\", \"fizzbuzz\"]]}]}" \
-H X-Scope-OrgId:foo
```

로그 조회:

```
curl "http://127.0.0.1:3100/loki/api/v1/query_range" --data-urlencode 'query={job="test"}' -H X-Scope-OrgId:foo | jq .data.result
```

• Loki에 전송된 로그를 쿼리하여 확인.
• X-Scope-OrgId 헤더를 사용해 특정 테넌트의 로그를 조회.

Grafana에서 Loki 연결
• Grafana Loki 데이터 소스 URL: 클러스터 내부에서 Grafana를 사용할 때 Loki Gateway의 DNS 주소.
• Grafana에 데이터 소스를 추가할 때 이 URL을 사용.

Multi-tenancy
• Loki가 멀티 테넌시(Multi-tenancy)를 지원하도록 구성됨.
• X-Scope-OrgID 헤더:
• 특정 테넌트를 식별하는 데 사용.
• 이 헤더가 없는 API 요청은 404 no org id 오류를 반환.

Grafana에서 Multi-tenancy 설정

    •	Grafana에서 Loki 데이터 소스를 추가할 때 HTTP Headers 섹션에 X-Scope-OrgID를 설정하여 특정 테넌트에 연결.

```
helm repo add grafana https://grafana.github.io/helm-charts


# Loki 설치 (auth disabled)
helm upgrade --install loki grafana/loki-stack --set auth_enabled=false

# Tempo 설치
helm upgrade --install tempo grafana/tempo --set server.log_level=debug

# Prometheus 설치
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus prometheus-community/prometheus

```

### loki

This is the second step.

### grafana

This is the second step.

{{% /steps %}}
