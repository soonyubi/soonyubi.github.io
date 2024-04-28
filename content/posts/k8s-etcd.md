---
title: "[k8s] 개념"
subtitle: ""
date: 2024-04-18T23:28:29+09:00
lastmod: 2024-04-18T23:28:29+09:00
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

hiddenFromHomePage: true
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

## etcd

etcd 명령을 실행하면, 2379 포트를 수신하는 서비스가 실행된다.
etcdctl은 etcd command line client 이고, key-value 쌍을 조회하거나 저장하는데 사용한다.
etcs에는 클러스터에서 사용하는 정보를 저장하는데, nodes/pods/configs/secrets/accounts/roles/bindings.. 을 저장한다.
클러스터에 변화를 주게되면, 모든 정보들은 etcd 서버에 업데이트 된다.

etcd 서버를 배포하는데에는 2가지 방법이 있는데, 처음부터 배포하는 것 그리고 kubeadm 을 이용해서 배포하는 것이 있다.

```
wget -q --https-only \
"https://github.com/coreos/etcd/releases/download/v3.3.9/etcd-v3.3.9-linux-amd64.tar.gz"
```

advertise-client-url 은 etcd 서버가 listen 하고 있는 주소이다.

kubeadm 을 이용해서 배포하면, etcd 서버는 포드로서 배포된다.

```
kubectl get pods -n kube-system
```

위 명령어를 이용해서, etcd 서비스가 배포되었는지 확인할 수 있따.

etcd 가 관리하고 잇는 key 리스트를 보기 위해선 다음 명령어로 확인할 수 있다.

```
kubectl exec etcd-master -n kube-system etcdctl get / --prefix -keys-only
```

고가용성을 위해, 여러개의 서버를 구성할 경우, etcd 가 여러 서버에 존재하게 되는데 모든 서버가 서로의 인스턴스에 대해 알게 해야한다.

## kube-api server

<p align='center'>
<img src="/images/k8s/cka/ch01/kubernetes-architecture.png" width="100%"/>
</p>

kubectl 명령을 입력하면, kube api server 가 요청에 대해 인증을 하고 유효성을 판단한 후 etcd 서버로 부터 원하는 응답을 리턴해준다.

kube api server는 etcd와 직접 소통하는 유일한 서버이다.

1. 유저를 인증
2. 요청의 유효성을 확인
3. 데이터를 검색하거나 업데이트
4. update etcd
5. scheduler
6. kubelet

## Kube Control Manager

Node Controller,Replication Controller 등의 수많은 패키지들이 모여있는 집합
Kube Control Manager 를 실행시키려면 다음 명령어를 통해 다운받고, kube-controller-manager.service를 실행하면 된다.
wget https://storage.googleapis.com/kubernetes-release/release/v1.13.0/bin/linux/amd64/kube-controller-manager

kube-controller-manager를 실행하게 되면 많은 옵션값들이 있는데 그 중 node-monitor-period/ node-monitor-grace-period/ pod-eviction-timeout 등등이 있따.
기본적으로 모든 컨트롤러 들이 활성화되지만, 몇몇개의 컨트롤러를 실행시킬지 말지 선택하는 옵션값도 있다.

위 방법 말고도, kubeadm 으로도 배포할 수 있다.

## kube scheduler

어떤 pod가 어떤 node 에 들어가야 할 지 결정만 함
wget https://storage.googleapis.com/kubernetes-release/release/v1.13.0/bin/linux/amd64/kube-scheduler

cat /etc/kubernetes/manifests/kube-scheduler.yaml
kubeadm 으로 배포했을 때 스케쥴러 옵션을 보는 명령어

## kublet

pod를 노드에 위치
노드와 파드를 모니터링

kubeadm 은 기본적으로 kublet을 자동으로 배포하지 않는다
따라서 worker node에 kublet을 설치해야 한다.

## kube proxy

pod network 는 쿠버네티스 클러스터 내에 위치한 내부 가상 망이다.
위 망을 이용해서 모든 노드들의 pod들이 서로 연결되어 있다.

kube proxy는 각각의 노드에 설치되어 실행되는 프로세스다
노드 내에 포드가 생성될 대 마다 kube proxy는 iptable을 업데이트하여 특정 트래픽이 특정 노드 내 포드로 갈 수 있도록 도와준다.

## pods

pod안에 한개의 컨테이너만 들어가는게 아님. 근데 스케일을 키울 경우, 같은 기능을 하는 컨테이너를 가지는 pod를 늘림.

pod 를 배포하는 방법
kubectl run nginx --image nginx
위 명령어는 노드 내에 pod를 생성하고 pod 내에는 도커 허브 레포지토리에서 받은 nginx 이미지가 실행되는 컨테이너를 설치한다.

pod를 배포하기 위해 yaml 파일을 작성하는데 상위 레벨에는 apiVersion, kind, metadata, spec을 정의한ㄷ.
apiVersion : 보통 pod를 작성할 때 쓰는 api Version 은 v1 을 쓴다.
kind: pod / service / replicaSet / deployment 등이 들어간다.
metadata : name / labels 의 하위 필드를 가지고, labels는 app이라는 하위필드를 가진다. name은 pod의 이름을 의미한다. label은 pod를 식별할 수 있는 필드값이다.
spec 에는 pod내에 위치하는 container가 실행하는 이미지를 쓴다. spec의 하위 필드인 container는 list형태이다 . 이유는 pod내에 여러개의 container가 실행될 수 있기 때문이다.

실생중인 pod 내 컨테이너 이미지를 교체하는 방법
kubectl edit ~~
.Yml 파일 수정

## replication controller

replication이 필요한 이유는 high availability. replication controller 는 포드가 정상적으로 실행되도록 보장한다. 부하가 증가해서 pod가 여러 노드에 걸쳐 증가하게 되면, replication controller는 부하를 분산하는 역할을 한다.

Replication controller 를 yaml 파일에 정의할 때 spec 필드 하위에 template필드가 존재하는데 template 필드에는 pod에 정의했던, label / spec이 들어간다. 추가로 replica set을 몇개 운용할 건지에 대한 정의를 하는 필드는 spec.replicas = #replica로 작성하면 된다.

현재 클러스터에 무슨 replication controller가 구성되었는지 확인하려면, kubectl get replicationcontroller 명령어를 사용하면 된다.

## Replica set

replica set을 yaml에 정의할 때 apiVersion = apps/v1 으로 정의한다. spec은 replication controller spec 에 정의한 것과 비슷하게 구성한다. replica set의 spec 필드에는 replication controller와 다른점이 있는데 selector 필드가 필요하다.( replication controller 에서도 사용 가능하다.)

Replica set 은 replication controller 의 다음버전으로, 더 세밀하게 파드 집합을 선택할 수 있다. 요즈음에는 replication controller보다는 replication Set이나 그보다 더 추상화된 deployment를 사용한다.

replicas의 숫자를 교체하는 방법에는 다음 방법이 있다.

1.replicaset-def.yaml 파일 수정 -> kubectl replace -f replicaset-def.yaml

2. kubectl scale --replicas=6 -f replicaset-def.yaml

pod container 이미지를 변경하고 kubectl 명령어로 적용해도 이미 실행중인 이미지는 바뀌지 않는다.
kubectl edit -f ~~.yml 로 변경하고 싶은 부분을 변경한 후, kubectl delete pods [...pods name] 을 통해서 모든 pod를 제거해준다.

## deployments

deployment는 Replicaset을 추상화 한 것으로, 배포 과정에서 배포를 중지하거나, 롤백하거나 등을 할 수 있다. definition.yml은 replicaset과 비슷하고, kind: Deployment로 해주면 된다.

## Services

백엔드/프론트엔드 포드간의 연결 또는 사용자와의 연결 또는 데이터 소스와의 연결을 제공한다.

웹앱을 배포했을 때 외부사용자가 어떻게 접근하는지?
ssh 로 노드에 접속해서, curl 명령어로 어플리케이션이 띄어진 pod의 내부 ip 로 요청을 보내면 되는데 이건 일반적인 방법이 아님. 그래서 우리는 service를 사용하는데, 특정 포트를 리슨하고 있다가 요청이 오게되면 해당 요청을 특정 pod로 요청해서 원하는 응답을 받을 수 있음

서비스의 유형

- nodePort: 서비스가 내부 포트를 노드의 포트에 엑세스할 수 잇게 해줌

<p align='center'>
<img src="/images/k8s/cka/ch01/service_nodeport.png" width="100%"/>
</p>

노드 자체에 외부에서 접근이 가능한 포트가 있음 이름 nodePort라고 함. nodeport 는 30000~32767의 값을 갖고 있음.
서비스는 이 nodePort를 리슨하고 있고 서비스 포트(port)와 pod의 내부포트(targetPort)를 연결함

서비스 yaml의 spec에는 type, ports 가 있음
위 그림과 똑같이 구성한다면,
Type: NodePort / ports.targetPort : 80, ports.port:80, ports.nodePort:30008 이 됨
ports는 array 타입임

서비스와 pod를 연결하는 값도 설정해줘야 됌.
pod에는 label이 부여되었을테니, selector를 이용해서 해당 pod 정보를 서비스 설정파일에 넣어주면 됨

부하 분산 목적으로 node 내에 여러개의 pod가 생성되어 있다면, 각각의 pod는 똑같은 label을 가지고 있어야하고 그 label을 service값에 넣어주면됨

만약 여러개의 node에 분산하게 되면, 쿠버네티스는 서비스를 이용해서 여러 노드에 걸쳐 해당 어플리케이션에 접근 가능하도록 관리한다.

- clusterIp: 내부 포드간에 commnuication 을 가능하게 함

모든 pod는 ip가 할당되어 있다. 이 IP들은 정적이 아님. pod가 제거되거나 추가되면 바뀌기 때문에 의존할 수 없다.

쿠버네티스는 서비스를 통해, pod를 그룹으로 짓고 단일 접근가능한 단일 interface를 제공한다.
이 단일 진입점을 clusterIP라고 한다. 여러 계층을 구성된 어플리케이션이라면, 프론트엔드에선 백엔드로 접근하기 위해선, 백엔드의 clusterIP로 접근해야 한다.

clusterIP를 생성하기 위해선, spec.type: ClusterIP를 생성하고, selector로 pod.label에 지정된 값들을 가져온다. 추가로, spec에는 pod의 노출 포트인 targetPort / 서비스의 포트인 port 값을 지정해줘야 한다.

- load balancer

## namespaces

각각의 namespace에는 policy를 정의할 수 있다.
다른 서비스의 어플리케이션에 접근하려면 다음과 같이 구성해야됨
service-name.namespace.service.domain

다른 namespace에 위치한 pod를 열거하려면,
kubectl get pods --namespace= ?

특정 pod를 정의할 때 namespace를 정의하려면 , metadata.namespace에 정의하면 된다.

namespace 자체를 생성하려면
kubectl create namespace []

특정 namespace를 현재 context로 옮겨오기 위해선(--namespace option 없이 쓰려면)
kubectl config set-context $(kubectl config current-context) --namespace=??

모든 namespace 에 위치한 pod를 보려면
kubectl get pods --all-namespace

namespace에서 리소스 할당량을 제어하려면, quota 파일을 생성하면 된다.

## Scheduler

스케쥴러는, pod를 생성할 때, nodeName이란 걸 쿠버네티스가 자동으로 지정해줌.
스케쥴러가 없다면, pod는 pending 상태임. 그래서 pod를 생성할 때 nodeName필드를 지정해줘야됌
nodeName은 생성 시에만 할당할 수 없음. 이미 생성된 pod의 nodeName은 변경못함
nodeName을 지정해주고 나면, binding api 에 생성할 binding 객체에 대한 정보를 json 포맷으로 전송해야됌

## labels and selector

label 에 env=dev라고 지정한 pod를 검색하는 방법
kubectl get pods --selector env=pod

label에 env=prod 인 모든 리소스의 갯수를 확인하는 방법
kubectl get all --selector env=prod --no-headers | wc -l

selector 로 여러개의 label을 검사하는 경우
kubectl get all --selector env=prod,bu=finance,...

## taint and tolerant

pod가 node에 배치될 때 어떤 제약을 가지고 배치되는 것을 의미
taint는 특정 label을 가진 pod만 배치되게 하는 거고, tolerant 는 taint를 가진 노드에 pod가 배치되려고 할 때 내성을 줘서 특정 taint를 가진 node라도 pod가 배치될 수 있게 하는 것을 의미

node에 taint를 추가하는 코드

kubectl taint nodes [node-name] key=value:[NoSchedule, PreferNoSchedule, NoExecute]

만약 kubectl taint node node1 color=blue:NoSchedule 명령을 이용해, node1에 color=blue라는 taint를 생성했다면,
해당 노드에 내성을 가지는 pod는 다음과 같이 생성할 수 있다.

```yaml
...
spec
  tolerations:
  - key: "color"
    operator:"Equals"
    value: "blue"
    effect: "noSchedule"
```

여기서 주의할 점은, double quote로 값을 입력해야 된다는 것이다.

쿠버네티스 클러스터가 구성되면, 쿠버네티스는 master node에 그 어떤 pod도 접근하지 못하도록 taint를 부여한다.

node에서 taint 제거하기
kubectl taint node controlplane node-role.kubernetes.io/control-plane:NoSchedule`-`

## Node selector & node affinity

kubectl label nodes [node-name] [key=value] 로 node에 label을 부여하게 되면,
pod가 스케쥴러에 의해 node에 배치될 때 특정 label을 가진 node에만 할당되게 할 수 있다.
spec.nodeSelector -> 노드에 지정한 key-value 쌍

node selector는 근데 한계가 있음. 예를 들어, 특정 label이 아닌 node에만 할당한다거나 할 때는 node selector로 구현할 수 없음

그래서 생겨난게 node affinity

node affinity를 사용하면 여러 expression을 사용할 수 있음,

만약, 특정 label에 대한 선호도를 가진 노드로 pod를 할당하라고 정의한 후, 나중에 해당 node의 label을 변경하면 어덯게 될까?
또는 특정 label이 존재하지 않는다면??

그러한 상황에 놓였을 때 pod가 어떤 행동을 할지 결정해주는 것이 node affinity types이다

node affinity type

pod의 lifecycle에는 2가지 상태가 있다. during-scheduling / during-execution

- requiredDuringSchedulingIgnoredDuringExecution

during scheduing 이 required이기 때문에, 특정 label을 가진 node를 찾지 못할경우, pod는 스케쥴되지 않는다.
during execution이 ignored이기 때문에, node에 할당된 특정 label을 지울경우, 해당 label에 대한 선호도를 가진 pod가 실행중인 상태일 경우 해당 pod가 계속 실행되도록 놔둔다.

- preferredDuringSchedulingIgnoredDuringExecution

during scheduing 이 preferred이기 때문에, 특정 label을 가진 node를 찾지 못해도 pod는 스케쥴링된다.

- requiredDuringSchedulingRequiredDuringExecution

특정 label에 대한 affinity 를 갖고 있는 pod가 node에 배치되고 실행중인 상태일 때, 특정 label을 node에서 제거했을 경우,
실행중인 pod라도 해당 pod를 node에서 제거한다.

## Resource requirement and limits

spec.resources 필드를 이용해서, pod가 실행되는데 필요한 리소스가 얼마나 필요한지 요청할 수 있다.
또한, spec.resources.limit필드에 pod가 실행될 때 리소스를 얼만큼 제한하는 지를 명시할 수 있다.

```yaml
spec:
  resources:
    requests:
      cpu: 2
      mem: "4Gi"
    limits:
      cpu: 2
      mem: "4Gi"
```

cpu는 Limit 에 명시된 것 이상 사용할 수 없지만, 메모리는 가능하다. 메모리가 초과하게 되면 oom 에러(out of memory)로 pod는 종료된다.

resource의 명세를 정의하지 않는다면, node 내에 실행되는 여러 어플리케이션은 서로를 질식시킬 수 있다.

특정 namespace 내에서 생성되는 pod의 리소스를 제한하기 위해산 resource quota를 사용한다. 이걸 사용하면, namespace간의 자원을 효율적으로 클러스터내에서 분배하여 사용할 수 있다.

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: example-quota
  namespace: example-namespace
spec:
  hard:
    requests.cpu: "1"
    requests.memory: 1Gi
    limits.cpu: "2"
    limits.memory: 2Gi
    pods: "10"
    services: "5"
    persistentvolumeclaims: "4"
    request.storage: 10Gi
```

### Pod 수정하기

Pod의 몇 가지 특정 필드만 직접 수정할 수 있어:

spec.containers[*].image
spec.initContainers[*].image
spec.activeDeadlineSeconds
spec.tolerations
그 외의 다른 설정들은 직접 수정할 수 없으며, 변경을 원할 경우 Pod를 삭제하고 수정된 설정을 반영하여 새로운 Pod를 생성해야 해. 이를 위해 두 가지 방법을 사용할 수 있어:

직접 수정 시도: kubectl edit pod <pod name> 명령어를 사용해 vi 에디터에서 Pod의 설정을 열어 수정을 시도할 수 있지만, 허용되지 않는 필드를 수정하려 하면 저장이 거부될 거야. 수정된 파일은 임시 위치에 저장되며, 기존 Pod를 삭제한 후 이 임시 파일을 사용해 새 Pod를 생성할 수 있어.
파일 추출 및 수정: kubectl get pod <pod name> -o yaml > my-new-pod.yaml 명령을 통해 Pod의 설정을 YAML 파일로 추출하고, vi 에디터를 사용해 필요한 수정을 한 후 저장해. 그 다음 기존의 Pod를 삭제하고 수정된 파일로 새로운 Pod를 생성해.

### Deployment 수정하기

Deployment의 경우, Pod 템플릿이 Deployment 스펙의 하위 요소로 포함되어 있기 때문에, Deployment 내의 Pod 템플릿에 대한 필드는 자유롭게 수정할 수 있어. 수정이 이루어지면 Deployment는 자동으로 변경 사항을 반영하여 기존 Pod를 삭제하고 새로운 Pod를 생성해. Deployment를 수정하려면 kubectl edit deployment <deployment name> 명령어를 사용하면 돼.

## Daemon set

daemon set은 노드가 새로 생성될 때 반복적으로 배포해야될 pod를 미리 생성해놓는것
예를 들어, monitoring / log collector / kube proxy 같은 것들 ..

daemon set을 생성하는 코드는 replicaSet 이랑 거의 똑같은데 kind만 다름

모든 namespace에 정의된 daemon set의 개수구하기
kubectl get daemonset -A

## static pod

만약 masternode가 존재하지 않을경우, kublet은 노드를 독립적으로 관리할 수 있다.
kubeapi server가 존재하지 않을 때, kublet은 /etc/kubernetes/manifests 파일에 위치한 pod definition 파일을 읽어서 pod를 노드내에 생성할 수 있다. kublet은 해당경로를 주기적으로 읽어서 pod를 삭제하거나 생성할 수 있다. 이런 pod를 static pod라고 한다.
해당 경로를 생성하는 방법은, kublet bin 파일을 이용해서 kublet 서비스를 실행할때 --pod-manifest-path 필드에 적절한 값을 주어서 경로를 생성할 수 있다.

또는 --config=kubeconfig.yaml 옵션을 주고 , kubeconfig yaml 파일에는 static pod 가 위치할 path를 적을 수 있다. (staticPodPath : /etc/kubernetes/manifest)

master node가 존재하고 kubeapi server 가 있을경우, http 통신을 사용해서 kublet이 static pod를 생성하게 할 수 있다.
이때 생성된 pod는 제거하거나 수정할 수 없다. kubeapi server를 이용해서 간접적으로 pod의 상태를 확인할 수 있다.

주로 중요한 pod를 생성할 때 static pod를 활용한다. 예를 들어, master node에 속해있는 kubeapi server / controller manager / scheduler등이 static pod이다.

- static pod vs daemon set

<p align='center'>
<img src="/images/k8s/cka/ch01/staticpodVSdaemon.png" width="100%"/>
</p>

How many static pods exist in this cluster in all namespaces?

일반적으로 kubectl get pods -A 로 모든 pod를 조회할 수는 있지만, 해당 pod가 static pod인지는 정확히 알 수 없다.
정확히 알기 위해선, pod를 describe 하고 해당 pod의 owner가 Node인지 확인해야 한다.

추가로, static pod의 경우, suffix로 해당 node의 이름이 붙기 때문에 특정 노드의 뒤에 node이름이 붙어있다면 해당 pod가 static pod인지 의심할 수 있다.

<p align='center'>
<img src="/images/k8s/cka/ch01/identify_staticPod.png" width="100%"/>
</p>

## multiple scheduler

특정 pod가 customize 된 스케쥴러에 의해 노드로 배치받길 원한다면, custom scheduler를 생성할 수 있다.

https://kubernetes.io/ko/docs/tasks/extend-kubernetes/configure-multiple-schedulers/

특정 pod가 custom scheduler에 할당되었는지를 확인하려면
kubectl get events -o wide를 통해 확인할 수 있다.

특정 scheduler에 문제가 생기면
kubectl log [scheduler-name] 을 통해 확인할 수 있다.

## scheduler profile

스케쥴러는 다음과 같은 프로세스로 운영된다.

pod-definition파일을이용해 Pod를 생성했다고 가정하자. 이때 priorityClassName을 통해 pod가 생성될 때 priority를 부여할 수도있다.

1. scheduling queue 에 생성될 pod가 들어간다.

- scheduling plugin : priority sort

2. filter step : queue에서 pod를 pop한 후, 들어갈 수 없는 노드들을 filtering 한다.

- scheduling plugin : Node resource fit / NodeUnscheduled(--dry-run) / NodeName

3. scoring step : 남은 node 중에서, 해당 pod를 할당했을 때 여유분이 많은 노드에 더 높은 가중치를 부여한다.

- scheduling plugin : Node resource fit plugin을이용해서 노드에 가중치를 부여
  image locality : pod가 가지고 있는 이미지와 같은 이미지인 node에게 높은 가중치를 부여

4. binding step: 가중치가 제일 높은 node에 pod를 할당한다.

- scheduling plugin : default binding

## Monitor

kubelet에는 서브 component로 cAdvisor 가 있는데 노드내에서 실행중인 포드의 메트릭을 받는다. 이 메트릭을 kubeapi server 를 이용해서 포드의 상태를 모니터링 할 수 있다.

minikube cluster를 운영중이라면 다음 명령어로 cAdvisor를 활성화 시킬 수 있다.
minikube addons enable metric-server

minikube를 운영하는 게 아니라면, metric-server.git 을 다운받고 다운받은 파일의 definition 파일을 적용해준다.

kubectl top node / pod 명령어를 이용해서 노드 또는 포드의 리소스 점유 상태를 확인할 수 있다.

## rolling update and rollback

pod의 이미지를 업데이트 하는방법에는 2가지가 있다.

1. yaml파일을 수정한 후 apply 명령을 적용
2. kubectl set image [deployment-name] [image]

이미지 변경을 적용하고 describe 해보면, strategyType : Recreate / Rolling update 방식 2가지가 있는 것을 볼 수 있다.
recreate 의 경우 실행되고 있는 pod를 한번에 종료후 바뀐 이미지가 적용된 컨테이너를 새로 올리는 것이고
rolling update의 경우 pod1개씩 종료하고 1개씩 새로 올리는 것이다.

쿠버네티스는 deployment를 새로 배포할 때 기존에 존재하는 replicaset을 구성하는 pod갯수만큼을 가지고 있는 replicaset을 새로 하나 생성하고 기존 replicaset의 pod를 죽이면서 새로운 replicaset에 새로운 pod를 띄운다.

kubectl rollout status deployment/my-deployment -> 배포의 상태를 모니터링할 수 있다.

rollback을 하기 위해선 다음 명령어를 실행해주면 된다.
kubectl rollout undo [deployment-name]

## commands and arguments

## environment variables

spec.container.env 는 key-value 여러개를 받아서 컨테이너가 실행될 때 환경변수를 등록하게 할 수 있다.
또한 단순히 key-value 뿐만 아니라, configMapRef / secretValueRef를 통해 참조할 수도 있다.

- config Maps

1. imperative way

```shell
kubectl create configMap \
    <config-name> --from-literal=<key>=<value>
                  --from-literal=<key>=<value>

kubectl create configMap \
    <config-name> --from-file=
```

2. declarative way

```shell
# configmap-definition.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: game-demo
data:
  # property-like keys; each key maps to a simple value
  player_initial_lives: "3"
  ui_properties_file_name: "user-interface.properties"

  # file-like keys
  game.properties: |
    enemy.types=aliens,monsters
    player.maximum-lives=5
  user-interface.properties: |
    color.good=purple
    color.bad=yellow
    allow.textmode=true


~ > kubectl create -f configmap-definition.yaml
```

<p align='center'>
<img src="/images/k8s/cka/ch01/configmap.png" width="100%"/>
</p>

## configure secret

## os upgrade

노드를 유지보수를 위해 빈 상태로 만들고, 다른 애플리케이션들이 해당 노드에 스케줄되지 않도록 설정하려면, 쿠버네티스에서 몇 가지 단계를 거쳐야 해. 이 과정을 "노드 드레이닝(draining)"이라고 하며, 파드들을 안전하게 다른 노드로 이동시키고 해당 노드를 유지보수 모드로 설정할 수 있어.

1. 노드를 Unschedulable로 설정하기
   먼저, 노드를 unschedulable 상태로 설정해서 새로운 파드가 스케줄되지 않도록 해야 해. 이것은 kubectl cordon 명령어로 할 수 있어:

bash
Copy code
kubectl cordon node01
이 명령어는 node01이라는 노드에 더 이상 새로운 파드가 할당되지 않도록 설정해.

2. 기존 파드들을 다른 노드로 이동시키기
   이제 node01에 있는 기존 파드들을 안전하게 다른 노드로 이동시킬 차례야. 이 작업은 kubectl drain 명령어를 사용해서 수행할 수 있어:

bash
Copy code
kubectl drain node01 --ignore-daemonsets --delete-emptydir-data
--ignore-daemonsets: 데몬셋으로 생성된 파드는 무시하고 드레인을 진행해야 하기 때문에 이 옵션을 추가해야 해.
--delete-emptydir-data: EmptyDir 볼륨을 사용하는 파드가 있다면, 이 데이터를 삭제하고 파드를 다른 노드로 이동시키기 위해 이 옵션을 사용해. 3. 유지보수 후 노드 복구
유지보수 작업이 끝나고 노드를 다시 사용 가능한 상태로 되돌리고 싶다면, 다음 명령어로 노드를 다시 schedulable 상태로 설정할 수 있어:

bash
Copy code
kubectl uncordon node01
이 명령어는 node01 노드를 다시 활성화시켜서 새로운 파드들이 스케줄될 수 있도록 해.

이런 단계를 통해 노드를 안전하게 유지보수 모드로 전환하고, 작업 후 다시 정상적으로 활성화할 수 있어! 필요한 도움이나 추가 질문이 있으면 언제든지 물어봐 줘.

## Cluster upgrade

controlplane 을 구성하는 각각의 pod의 버전은 kube-apiserver를 넘어설 수 없다.

master node를 업그레이드 하기 위해 다운시켜도, worker node가 운영되는 것에는 지장이 없다.
하지만, 새로운 어플리케이션을 배포하는 것은 불간으하다.

## backup and restore

https://velog.io/@khyup0629/K8S-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0-%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98-%EC%84%A4%EC%B9%98-%EB%B0%8F-%EC%84%A4%EC%A0%95

## restore etcd

```shell
Command:
      etcd
      --advertise-client-urls=https://192.6.229.9:2379
      --cert-file=/etc/kubernetes/pki/etcd/server.crt
      --client-cert-auth=true
      --data-dir=/var/lib/etcd
      --experimental-initial-corrupt-check=true
      --experimental-watch-progress-notify-interval=5s
      --initial-advertise-peer-urls=https://192.6.229.9:2380
      --initial-cluster=controlplane=https://192.6.229.9:2380
      --key-file=/etc/kubernetes/pki/etcd/server.key
      --listen-client-urls=https://127.0.0.1:2379,https://192.6.229.9:2379
      --listen-metrics-urls=http://127.0.0.1:2381
      --listen-peer-urls=https://192.6.229.9:2380
      --name=controlplane
      --peer-cert-file=/etc/kubernetes/pki/etcd/peer.crt
      --peer-client-cert-auth=true
      --peer-key-file=/etc/kubernetes/pki/etcd/peer.key
      --peer-trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
      --snapshot-count=10000
      --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt

```

위 옵션 중 url 관련 옵션
--advertise-client-urls=https://192.6.229.9:2379
--initial-advertise-peer-urls=https://192.6.229.9:2380
--initial-cluster=controlplane=https://192.6.229.9:2380
--listen-client-urls=https://127.0.0.1:2379,https://192.6.229.9:2379
etcd 서버가 클라이언트로 부터 요청을 받기 위해 특정 주소에서 리스닝하고 있는 url을 의미한다.
여기서 localhost는 etcd 서버가 실행되고 있는 머신과 같은 머신에서 실행되고 있는 서버의 주소를 받는 url을 의미하고
뒤 주소는 etcd 서버가 설치된 머신의 네트워크 인터페이스에 할당된 주소를 의미한다. 클라이언트나 클러스터내 다른 pod는 해당 주소를 통해 etcd 서버에 접속할 수 있다.
--listen-metrics-urls=http://127.0.0.1:2381
--listen-peer-urls=https://192.6.229.9:2380
