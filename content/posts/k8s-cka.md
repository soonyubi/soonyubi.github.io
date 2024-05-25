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

## SCHEDULER

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

## MONITORING

### 리소스 사용량 모니터링

kubectl top node 명령어를 통해서 노드의 cpu , memory 사용량을 확인할 수 있다.
kubectl top pod 명령어를 통해서 pod 의 cpu, memory 사용량을 확인할 수 있다.

kubectl logs [pod-name] -c [container-name] 명령어를 통해, pod에서 운영중인 컨테이너의 로그를 확인할 수 있다.

## APPLICATION LIFECYCLE

### rolling updates and rollback

kubectl describe deployment 를 통해 정보를 확인해보면, `strategyType`필드가 존재한다. 이 필드에는 2가지의 값이 있는데 다음과 같다.

- rolling update : 서비스의 중단 없이 이전버전의 인스턴스를 줄이면서, 새로운 버전의 인스턴스를 늘리는 방식이다.
- recreate : 기존 pod를 먼저 제거하고, 새로운 pod를 생성하는 방식이다.

### Commands and arguments

```Dockerfile
FROM python:3.6-alpine

RUN pip install flask

COPY . /opt/

EXPOSE 8080

WORKDIR /opt

ENTRYPOINT ["python", "app.py"]

CMD ["--color", "red"]
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: webapp-green
  labels:
    name: webapp-green
spec:
  containers:
    - name: simple-webapp
      image: kodekloud/webapp-color
      command: ["--color", "green"]
```

위 2개의 컨테이너 명세서를 작성할 때, startup 시점에 실행되는 명령어는 ??
Dockerfile에서 명령어를 보면, entrypoint 는 컨테이너의 실행파일을 설정하고, cmd는 전달 인자를 지정하여 `python app.py --color red` 명령어가 실행된다.

pod를 생성하는 yaml파일을 보면, command를 지정하여, Dockerfile의 Entrypoint 가 overriding 되어 dockerfile에서 설정한 명령어는 실행되지 않는다.

따라서 실행시점에 실행되는 명령어는 `--color green`이 된다.

```yaml
---
apiVersion: v1
kind: Pod
metadata:
  name: webapp-green
  labels:
    name: webapp-green
spec:
  containers:
    - name: simple-webapp
      image: kodekloud/webapp-color
      args: ["--color", "green"]
```

위와 같이 argument만 정해도 컨테이너가 실행이 될까 ?? argument만 정의해주게 되면 컨테이너 이미지 ENTRYPOINT의 인자로 전달되어 컨테이너를 실행시킬 수 있다.

### Configmap

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: configmap-demo-pod
spec:
  containers:
    - name: demo
      image: alpine
      command: ["sleep", "3600"]
      env: # 환경변수는 여러개를 등록할 수 있다.
        - name: PLAYER_INITIAL_LIVES # 컨테이너에서 해당 이름을 가지고 환경변수에 접근할 수 있다.
          valueFrom: # 쿠버네티스 리소스로 부터 가져오는 것을 의미한다.
            configMapKeyRef:
              name: game-demo # config map은 동일한 네임스페이스에 위치해야 하고, 미리 생성되어 있어야 한다.
              key: player_initial_lives # config map안에서 어떤 값을 가져올지를 결정한다.
        - name: UI_PROPERTIES_FILE_NAME
          valueFrom:
            configMapKeyRef:
              name: game-demo
              key: ui_properties_file_name
```

위와 같이 구성하게되면, pod 내의 container에서 `PLAYER_INITIAL_LIVES` / `UI_PROPERTIES_FILE_NAME` 필드를 통해 config map에 정의한 value을 읽어올 수 있게 된다.

### secrets

쿠버네티스 리소스 secret 을 정의하고 나면, 컨테이너에서 다음과 같이 사용할 수 있다.

```yaml
...
# 여러개의 secret값을 환경변수로 한번에 넘기는 방법
containers:
  - name: envars-test-container
    image: nginx
    envFrom:
    - secretRef:
        name: test-secret

# 한개씩 환경변수로 넘기는 방법
containers:
  - name: envars-test-container
    image: nginx
    env:
    - name: BACKEND_USERNAME
      valueFrom:
        secretKeyRef:
          name: backend-user
          key: backend-username
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: db-user
          key: db-username
```

### default namespace가 아닌, 다른 namespace에 위치한 pod의 로그를 확인하기

```bash
kubectl -n [other-namespace] exec -it [pod-name] cat [log file directory]
```

### multi container - sidecar container를 추가하여, 로그 파일을 사이드카에 mount 하기

<p align='center'>
<img src="/images/k8s/cka/ch01/kubernetes-ckad-elastic-stack-sidecar_l4d1ga.png" width="100%"/>
</p>

위 그림처럼, 구성을 하고자 specification을 다음과 같이 정의했다.

```yaml
containers:
  - image: kodekloud/filebeat-configured
    imagePullPolicy: Always
    name: sidecar
    resources: {}
    terminationMessagePath: /dev/termination-log
    terminationMessagePolicy: File
    volumeMounts:
      - mountPath: /var/log/event-simulator/
        name: log-volume
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-t6mms
        readOnly: true
  - image: kodekloud/event-simulator
    imagePullPolicy: Always
    name: app
    resources: {}
    terminationMessagePath: /dev/termination-log
    terminationMessagePolicy: File
    volumeMounts:
      - mountPath: /log
        name: log-volume
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-t6mms
        readOnly: true
---
volumes:
  - hostPath:
      path: /var/log/webapp
      type: DirectoryOrCreate
    name: log-volume
```

위 처럼 구성을 하게 되면, sidecar, app container는 log-volume을 이용해 서로다른 컨테이너에서 같은 공간을 공유할 수 있게 된다. mountPath는 컨테이너 내에서 해당 볼륨이 마운트되는 위치를 의미한다.

### init container

init container는 메인 container가 실행되기 전에, 데이터 로드, 설정 파일 준비, 데이터베이스 마이그레이션, 네트워크 설정같은 작업을 수행하기 위해 사용하는 컨테이너이다.

init container가 정의되어 있는 경우, main 컨테이너가 실행되기 전에 먼저 실행됨을 보장하고, main 컨테이너와는 독립적으로 실행되어 영향을 주지 않는다. init container는 main container와 동일한 네트워크 또는 리소스를 사용할 수 있어 init container가 데이터를 다운로드하고 main container가 다운받은 데이터를 사용할 수 있다.

## CLUSTER MAINTENANCE

### OS upgrade

특정 node의 유지보수를 위해, node에 할당된 pod를 제거해야 하는 경우가 존재할 수 있다. 그러기 위해선 다음과 같은 단계를 따라야 한다.

1. 노드를 unschedulable 상태로 변경

```bash
kubectl cordon [node-name]
```

위 명령어를 통해 해당 노드에 pod가 스케쥴링 되지 않도록 할 수 있다.

2. 노드에서 모든 pod를 제거

```bash
kubectl drain [node-name]  --ignore-daemonsets --delete-local-data
```

위 명령어를 통해 노드에 할당된 pod를 모두 제거할 수 있다.
--ignore-daemonsts 옵션을 주는 이유는, 노드에 할당된 daemon set은 특정 용도로 노드에 존재하기 때문에 제거하지 않을 수 있다.
--delete-local-data 옵션을 주는 이유는, pod가 local data를 가지고 있는 경우, 이를 제거하고 pod를 다른 노드로 이동할 수 있게 한다. 이 때 local data 가 날아갈 수 있음에 주의해야한다.

3. 유지보수가 끝나면

```bash
kubectl uncordon [node-name]
```

위 명령어를 통해 노드에 pod가 다시 스케쥴링 될 수 있도록 구성한다.

### kubectl drain [node] 명령이 실패하는 경우

replica set / stateful set / daemon set 에서 관리하는 pod 가 아닌 pod가 node에 존재할 때 kubectl drain 명령을 실행하게 되면 실패하게 된다. 그 이유는 해당 pod 가 `orphan` pod 가 되기 때문인데, 이 pod는 컨트롤러에 의해 다른 노드에 배치되지 않기 때문이다.

replica set / stateful set / daemon set 으로 생성된 pod만이 존재하는 경우 kubectl drain 명령은 성공적으로 수행된다. 그 이유는 컨트롤러가 관리하고 있기 때문이다.

만약, --force 옵션으로 강제로 drain 시키게 되면, 컨트롤러에 의해 관리되지 않는 pod를 영원히 유실할 수 있음에 주의해야 한다.

### cluster upgrade

```bash
kubectl version
```

위 명령어를 통해, 현재 kubectl의 버전을 확인할 수 있다.

```bash
kubeadm upgrade plan

Upgrade to the latest version in the v1.28 series:

COMPONENT                 CURRENT   TARGET
kube-apiserver            v1.28.0   v1.28.10
kube-controller-manager   v1.28.0   v1.28.10
kube-scheduler            v1.28.0   v1.28.10
kube-proxy                v1.28.0   v1.28.10
CoreDNS                   v1.10.1   v1.10.1
etcd                      3.5.9-0   3.5.9-0
```

위 명령어를 사용해서, 현재 사용중인 버전 대비 업그레이드 할 수 있는 버전을 확인할 수 있다.

### controlplane node 업그레이드

1. kubernetes.list 파일 수정

kubernetes.list 파일은 kubernetes 관련 패키지를 관리하는 저장소의 위치를 정의하는 공간이다. 이를 통해 쿠버네티스 패키지를 어디서 찾아야 할 지 apt 패키지 매니저가 알 수 있다.

2. apt 패키지 매니저 갱신

```bash
apt update # 최신 패키지 정보를 포함하는 로컬 데이터베이스를 갱신한다.
apt-cache madison kubeadm # kubeadm 패키지에 대해 사용가능한 버전과 저장소 위치를 표시한다.

# 출력내용
# kubeadm | 1.29.5-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
# kubeadm | 1.29.4-2.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
# kubeadm | 1.29.3-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
# kubeadm | 1.29.2-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
# kubeadm | 1.29.1-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
# kubeadm | 1.29.0-1.1 | https://pkgs.k8s.io/core:/stable:/v1.29/deb  Packages
```

3. kubeadm 최신 버전 설치, 적용

위 출력내용을 토대로, 1.29 버전을 설치할 수 있기 때문에 다음과 같은 명령어를 통해 kubeadm을 업그레이드 시켜준다.

```bash
# controlplane upgrade
apt-get install kubeadm=1.29.0-1.1

kubeadm upgrade plan v1.29.0
kubeadm upgrade apply v1.29.0

apt-get install kubelet=1.29.0-1.1
systemctl daemon-reload
systemctl restart kubelet
kubectl uncordon controlplane

# node upgrade
ssh node

apt-get install kubeadm=1.29.0-1.1
kubeadm upgrade node

apt-get install kubelet=1.29.0-1.1
systemctl daemon-reload
systemctl restart kubelet
```

### backup and restore

etcd 서버를 백업하고 복구하는 방법

```bash
# backup
ETCDCTL_API=3 etcdctl --endpoints=https://[127.0.0.1]:2379 \
--cacert=/etc/kubernetes/pki/etcd/ca.crt \
--cert=/etc/kubernetes/pki/etcd/server.crt \
--key=/etc/kubernetes/pki/etcd/server.key \
snapshot save /opt/snapshot-pre-boot.db

# restore
ETCDCTL_API=3 etcdctl --data-dir /var/lib/etcd-from-backup snapshot restore /opt/snapshot-pre-boot.db

# etcd server 를 생성하는 manifest 파일 수정 --data-dir 파일 값을 위에 적어준 값으로 수정
vi /etc/kubernetes/manifest/etcd.yaml

```

/etc/kubernetes/manifest/etcd.yaml 파일이 변경되게 되면, etcd pod 는 static pod 이므로 자동으로 재생성된다.

### cluster 개수 확인하기

```bash
kubectl config get-clusters
```

### 여러개의 cluster 운영중일 때 context 변경하기

```bash
kubectl config use-context [cluster name]
```

### Etcd 서버가 stacked, external, 존재하지 않는지 확인하기

```bash
# context 변경
kubectl use-context [etcd 서버가 존재하는 cluster 이름]

# kube-system namespace 안에 존재하는 지 확인
# 여기에서 etcd 를 발견했다면, stacked
kubectl get pods -n kube-system

# 존재하지 않을 경우, controlplane에 접속
ssh controlplane

# etcd 구성 파일이 존재하는지 확인
ls /etc/kubernetes/manifests

# etcd 설정 파일이 없다면, 프로세스 목록에서 api server가 etcd 를 참조하고 있는지 확인
# 여기에서 etcd 외부 ip를 알 수 있음, 이를 통해 ssh 접속도 가능
ps -ef | grep etcd

# 여기에도 없으면 no etcd
```
