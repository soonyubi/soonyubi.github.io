---
title: "CKA 시험 정리본"
subtitle: ""
date: 2024-05-23T18:47:48+09:00
lastmod: 2024-05-23T18:47:48+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: []

featuredImage: ""
featuredImagePreview: ""

hiddenFromHomePage: false
hiddenFromSearch: false
twemoji: false
lightgallery: true
ruby: true
fraction: true
fontawesome: true
linkToMarkdown: true
rssFullText: false

toc:
  enable: true
  auto: true
code:
  copy: true
  maxShownLines: 50
math:
  enable: false
  # ...
mapbox:
  # ...
share:
  enable: true
  # ...
comment:
  enable: true
  # ...
library:
  css:
    # someCSS = "some.css"
    # located in "assets/"
    # Or
    # someCSS = "https://cdn.example.com/some.css"
  js:
    # someJS = "some.js"
    # located in "assets/"
    # Or
    # someJS = "https://cdn.example.com/some.js"
seo:
  images: []
  # ...
---

## BASIC CONCEPT

### pod 1라인 커맨드로 생성하기

```bash
kubectl run nginx-pod --image=nginx
```

### kubectl get pods 명령어를 실행했을 때 ready상태가 의미하는 것

`pod내에 실행되고 있는 컨테이너 수 / pod내에 총 컨테이너 수`

### replica set 으로 Pod를 배포할 때, pod를 삭제해도 재배포된다.

replica set은 specification 에 정의한대로, desired state의 수를 유지하려고 한다.

### 아래 spec에서 잘못된 점은?

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: replicaset-2
spec:
  replicas: 2
  selector:
    matchLabels:
      tier: frontend
  template:
    metadata:
      labels:
        tier: nginx
    spec:
      containers:
        - name: nginx
          image: nginx
```

replica set을 생성할 때, selector의 label과 template의 레이블은 동일해야 한다.
그래야 rs가 pod를 관리할 수 있기 때문이다.

### replica set의 scaling 변경

#### Scaling a ReplicaSet

kubectl scale 명령어를 통해서 수정할 수 있지만, kubectl edit 명령어로도 수정할 수 있다.

A ReplicaSet can be easily scaled up or down by simply updating the `.spec.replicas` field. The ReplicaSet controller ensures that a desired number of Pods with a matching label selector are available and operational.

When scaling down, the ReplicaSet controller chooses which pods to delete by sorting the available pods to prioritize scaling down pods based on the following general algorithm:

1. Pending (and unschedulable) pods are scaled down first
2. If controller.kubernetes.io/pod-deletion-cost annotation is set, then the pod with the lower value will come first.
   Pods on nodes with more replicas come before pods on nodes with fewer replicas.
3. If the pods' creation times differ, the pod that was created more recently comes before the older pod (the creation times are bucketed on an integer log scale when the LogarithmicScaleDown feature gate is enabled)
4. If all of the above match, then selection is random.

### 서로 다른 namespace 간 dns

예를 들어, blue pod 가 marketing namespace에 존재하고, db-service가 db-marketing namespace에 존재한다고 가정해보자.
blue pod가 다른 namespace에 위치한 db-service에 접근하기 위해선 아래와 같은 dns로 찾아야 한다.

<service-name>.<namespace>.svc.<cluster-domain>

따라서, db-service.db-marketing.svc.cluster.local의 도메인으로 db-service를 접근할 수 있다.

### 선언형으로 service 생성하기

아래와 같은 스펙으로 주어졌을 때

- Service: redis-service
- Port: 6379
- Type: ClusterIP

아래와 같은 명령어로 서비스를 생성할 수 있다.

```bash
kubectl create service clusterip redis-service --tcp=6379:6379
```

### 선언형으로, 특정 포트를 노출하는 pod 생성하기

아래와 같은 명령어를 작성하게 되면

```bash
kubectl run httpd --image httpd:alpine --port 80 --expose
```

httpd 이미지를 가진 pod와 80번 포트를 노출하는 서비스가 생성된다.

### pod를 실행시키지 않고, 선언 파일만 생성하기

```bash
kubectl run temp-pod --image nginx --dry-run=client -o yaml > pod.yaml
```

## scheduler

### scheduler서버가 다운되어 pod가 pending 상태로 남아있는 경우

scheduler 는 pod가 적절한 노드에 위치하게 하도록 도와준다. node의 자원, affinity, 등을 고려하여 pod를 적절히 노드에 배치한다.
만약 scheduler 서버가 존재하지 않아서 수동으로 Pod를 특정 노드에 위치하게 하려면 `spec.nodeName` 필드에 적절한 노드이름을 명시해야 한다.

### 특정 label을 가진 pod를 조회하는 방법

```bash
kubectl get pods --selector env=dev

또는

kubectl get pods --label-columns env,... 로 env 키를 가진 label을 조회한 후

app-2-qgnp2   1/1     Running   0          3m51s   prod
db-1-7gtv4    1/1     Running   0          3m51s   dev
db-1-rkzx9    1/1     Running   0          3m51s   dev
db-1-x9jvl    1/1     Running   0          3m51s   dev
app-1-9ggx2   1/1     Running   0          3m51s   dev
db-1-jm5s2    1/1     Running   0          3m51s   dev
auth          1/1     Running   0          3m51s   prod
app-1-7f7px   1/1     Running   0          3m51s   dev
app-1-g9f6z   1/1     Running   0          3m51s   dev
app-1-zzxdf   1/1     Running   0          3m50s   prod
db-2-fhv47    1/1     Running   0          3m51s   prod

출력되는 내용을 보고 개수를 세어줄 수 도 있다.
```

### 여러개의 label로 필터링하는 경우

```
kubectl get pods --selector env=prod,bu=finance,tier=frontend

여러개의 필터링을 걸때, 띄어쓰기를 하지 않아야함에 주의
```

### taint and toleration

특정 노드가 gpu를 가지고 있고, gpu를 가진 노드위에서만 돌려야 하는 pod 가 있다고 가정해보자. 우리는 특정 pod가 특정 node에만 위치하기 원하기 때문에 이 때 사용하는 것이 taint, toleration 이다.

taint는 노드에 적용되는 표식인데, NoSchedule, PreferNoSchedule, NoExecute의 표식이 있다.
노드에 taint를 적용하는 방법은 아래와 같다.

```bash
kubectl taint nodes node01 [key]=[vale]:[effect]
```

해당 노드에만 스케쥴링 되는 Pod를 생성하기 위해선 `spec.tolerations` 필드를 추가하기만 하면 된다.

### Node affinity

https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#operators

### Resource

특정 pod가 실패하는 이유를 찾을 때 유용한 명령어

```bash
kubectl describe pod temp | grep -A5 State

State 줄 이후에 5줄까지 보여준다.
```

OOMKilled -> 메모리가 충분하지 않아 실행하는데 실패했다는 의미이다.

### DaemonSet

#### DaemonSet의 주요 특징

- 자동 배포: 새로운 노드가 클러스터에 추가될 때, DaemonSet은 자동으로 해당 노드에 설정된 Pod를 배치한다. 반대로 노드가 제거되면, 해당 노드의 Pod도 제거된다.
- 노드 선택: DaemonSet은 특정 노드 또는 노드 그룹에만 Pod를 배치하도록 구성할 수 있다. 이는 노드 셀렉터나 노드 어피니티를 사용해 특정 라벨이 있는 노드에만 Pod를 배치하는 방식으로 수행될 수 있다.
- 자원 보장: 모든 노드에 항상 동일한 서비스가 실행되도록 보장해주기 때문에, 특정 데몬 또는 서비스가 클러스터의 모든 부분에서 사용 가능하도록 한다.

#### DaemonSet 사용 예

- 로깅 및 모니터링: 각 노드에서 로그를 수집하거나 시스템 지표를 모니터링하는 에이전트를 실행할 때, DaemonSet을 사용하면 모든 노드에서 동일한 로깅 또는 모니터링 도구가 실행됨을 보장할 수 있다.
- 네트워크 프록시: 각 노드에서 트래픽을 관리하거나 네트워크 규칙을 적용하는 프록시나 방화벽을 구현할 때 사용될 수 있다.
  보안: 보안 에이전트를 클러스터의 모든 노드에 설치해 각 노드의 보안을 강화하는 데에도 DaemonSet이 유용하다.

#### daemonset이 될 수 있는 컴포넌트들

- kube-flannel-ds
- kube-proxy
- cilium agent
- node exporter
- datadog agent

### static pod

kubelet이 직접 관리하는 pod이며, kubernetes api 서버를 통해 생성, 수정, 삭제되지 않는다.
kubelet이 특정 manifest를 읽어서 배포하는 pod이고, 해당 경로에 변경이 생기면 kubelet이 변경을 감지하고 자동으로 배포한다.

kubelet을 구성하는 파일은 /var/lib/kubelet/config.yaml에 정의되어 있으며, staticPodPath도 여기에 정의되어 있다.
보통은 /etc/kubernetes/manifests 밑에 staticPod를 정의한다.

static pod 맨 마지막 문제는 다시 풀어보기

### multiple scheduler

여러개의 scheduler를 운용할 때, 특정 pod가 특정 scheduler에 의해 배치되도록 할 수 있다.
pod definition 파일에 spec.schedulerName 에 scheduler를 정의해주면 된다.

### 리소스 사용량 모니터링

kubectl top node 명령어를 통해서 노드의 cpu , memory 사용량을 확인할 수 있다.
kubectl top pod 명령어를 통해서 pod 의 cpu, memory 사용량을 확인할 수 있다.

kubectl logs [pod-name] -c [container-name] 명령어를 통해, pod에서 운영중인 컨테이너의 로그를 확인할 수 있다.
