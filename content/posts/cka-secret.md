---
title: "Certificated Kubernetes Administrator - Secret"
subtitle: ""
date: 2024-05-26T15:21:59+09:00
lastmod: 2024-05-26T15:21:59+09:00
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

### configure certification file

```
# kubectl describe pods -n kube-system kube-apiserver-controlplane
Command:
      kube-apiserver
      --advertise-address=192.8.216.9
      --allow-privileged=true
      --authorization-mode=Node,RBAC
      --client-ca-file=/etc/kubernetes/pki/ca.crt
      --enable-admission-plugins=NodeRestriction
      --enable-bootstrap-token-auth=true
      --etcd-cafile=/etc/kubernetes/pki/etcd/ca.crt
      --etcd-certfile=/etc/kubernetes/pki/apiserver-etcd-client.crt
      --etcd-keyfile=/etc/kubernetes/pki/apiserver-etcd-client.key
      --etcd-servers=https://127.0.0.1:2379
      --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt
      --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
      --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
      --proxy-client-cert-file=/etc/kubernetes/pki/front-proxy-client.crt
      --proxy-client-key-file=/etc/kubernetes/pki/front-proxy-client.key
      --requestheader-allowed-names=front-proxy-client
      --requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt
      --requestheader-extra-headers-prefix=X-Remote-Extra-
      --requestheader-group-headers=X-Remote-Group
      --requestheader-username-headers=X-Remote-User
      --secure-port=6443
      --service-account-issuer=https://kubernetes.default.svc.cluster.local
      --service-account-key-file=/etc/kubernetes/pki/sa.pub
      --service-account-signing-key-file=/etc/kubernetes/pki/sa.key
      --service-cluster-ip-range=10.96.0.0/12
      --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
      --tls-private-key-file=/etc/kubernetes/pki/apiserver.key

#kubectl describe pods -n kube-system etcd-controlplane
Command:
      etcd
      --advertise-client-urls=https://192.8.216.9:2379
      --cert-file=/etc/kubernetes/pki/etcd/server.crt
      --client-cert-auth=true
      --data-dir=/var/lib/etcd
      --experimental-initial-corrupt-check=true
      --experimental-watch-progress-notify-interval=5s
      --initial-advertise-peer-urls=https://192.8.216.9:2380
      --initial-cluster=controlplane=https://192.8.216.9:2380
      --key-file=/etc/kubernetes/pki/etcd/server.key
      --listen-client-urls=https://127.0.0.1:2379,https://192.8.216.9:2379
      --listen-metrics-urls=http://127.0.0.1:2381
      --listen-peer-urls=https://192.8.216.9:2380
      --name=controlplane
      --peer-cert-file=/etc/kubernetes/pki/etcd/peer.crt
      --peer-client-cert-auth=true
      --peer-key-file=/etc/kubernetes/pki/etcd/peer.key
      --peer-trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
      --snapshot-count=10000
      --trusted-ca-file=/etc/kubernetes/pki/etcd/ca.crt
```

kube-apiserver의 `--etcd-certfile` 옵션과 etcd server의 `--cert-file` 옵션의 값을 보면 서로 다른 것을 확인할 수 있다.
전자는, kube-apiserver가 etcd server와 통신할 때 사용하는 클라이언트 인증서를 의미하고 후자는 etcd 서버 자체의 인증서를 의미한다.

### What is the Common Name (CN) configured on the Kube API Server Certificate?

kube api server 의 ssl/tls 인증서에 설정된 '공통 이름'이 무엇인지를 물어보는 것이다. 'Common Name'은 인증서가 발급된 서버의 도메인 이름을 지정하는데 사용한다.
쿠버네티스 환경에서 CN은 kubeapi server가 사용하는 주소를 나타낸다.

```bash
openssl x509 -in /etc/kubernetes/pki/apiserver.crt -text -noout
Certificate:
    Data:
        Version: 3 (0x2)
        Serial Number: 4169429065381004709 (0x39dcc9a4f63945a5)
        Signature Algorithm: sha256WithRSAEncryption
        Issuer: CN = kubernetes
        Validity
            Not Before: May 26 06:08:28 2024 GMT
            Not After : May 26 06:13:28 2025 GMT
        Subject: CN = kube-apiserver
        Subject Public Key Info:
            Public Key Algorithm: rsaEncryption
                Public-Key: (2048 bit)
                Modulus:
                    00:a9:e4:03:72:0f:63:ee:57:c2:25:b7:ff:57:6a:
                    9f:78:37:07:52:11:48:98:d8:b4:a3:ca:4a:05:36:
                    52:af:0d:50:69:60:60:1d:b9:e7:1a:98:6f:e4:2b:
                    73:66:aa:9d:77:ce:f3:18:e7:b1:09:83:ec:bb:ce:
                    5f:fd:88:a1:48:7f:50:94:26:a1:4b:ef:12:bf:92:
                    25:ce:a5:8d:6c:bb:7a:3e:6f:f5:78:2f:47:8a:e3:
                    18:b9:14:11:69:f2:a4:0d:4c:f9:c7:1b:e2:b5:ab:
                    09:7b:7b:3a:cf:37:bf:e6:8b:8e:2f:b1:89:0c:2a:
                    bc:2a:a1:14:b2:04:61:6c:6e:01:42:52:bb:c9:bf:
                    f0:a2:cf:53:fd:0a:be:64:34:1a:f0:93:84:57:4f:
                    f3:9a:b9:02:41:f9:21:3e:4a:a8:a4:4b:ca:41:7b:
                    8f:ed:98:b1:ef:91:58:20:ba:dd:74:45:a4:13:81:
                    ba:6f:7b:07:8d:d6:6d:bf:03:15:01:40:4a:e1:52:
                    1d:ba:a4:5e:80:1a:42:9f:0e:4a:b4:ef:0f:d0:5c:
                    51:c0:90:e6:44:e5:4e:03:7f:1c:85:6f:b6:26:23:
                    52:f7:e2:f8:b8:77:6f:81:36:93:ea:42:55:90:7a:
                    c0:7c:e4:f4:66:5b:a0:45:c2:e0:f3:99:53:1e:aa:
                    2b:95
                Exponent: 65537 (0x10001)
        X509v3 extensions:
            X509v3 Key Usage: critical
                Digital Signature, Key Encipherment
            X509v3 Extended Key Usage:
                TLS Web Server Authentication
            X509v3 Basic Constraints: critical
                CA:FALSE
            X509v3 Authority Key Identifier:
                F3:54:72:08:EE:F6:FC:9A:27:C9:61:F6:37:6D:32:B3:8B:86:A7:9B
            X509v3 Subject Alternative Name:
                DNS:controlplane, DNS:kubernetes, DNS:kubernetes.default, DNS:kubernetes.default.svc, DNS:kubernetes.default.svc.cluster.local, IP Address:10.96.0.1, IP Address:192.8.216.9
    Signature Algorithm: sha256WithRSAEncryption
    Signature Value:
        62:9d:43:f9:a8:db:73:5f:36:fe:8e:14:b4:00:9a:27:f5:7f:
        76:6a:ab:64:a4:65:7c:2a:ca:43:c7:6a:f3:fd:10:0c:a3:d5:
        f8:68:31:23:3a:0a:7e:10:51:1c:1d:9c:36:d0:05:2a:12:97:
        6e:8d:29:d4:9b:42:95:54:42:77:b6:cc:72:82:7e:98:f7:8b:
        8d:77:f9:cf:ae:4e:1a:d4:3c:a4:67:e5:94:1e:a7:d9:dd:8a:
        73:e4:d6:58:32:0e:4d:0e:8c:1b:51:a7:85:06:23:1c:ae:cf:
        30:2f:62:24:1c:1c:1b:a8:45:c9:69:59:08:17:c9:0d:b8:08:
        99:f4:01:05:76:2a:e7:4c:a2:f1:7e:71:a0:b6:6e:95:c5:55:
        3b:8c:67:a6:e1:e5:31:ca:2a:9d:b0:45:df:89:d0:77:2c:d9:
        9a:89:ee:35:99:ab:34:8e:b3:41:a7:f7:3a:6e:83:4f:a1:05:
        65:42:7b:db:91:9f:0f:55:36:ec:82:12:78:cf:bf:57:1a:95:
        31:18:f2:6e:5a:7b:61:11:cc:06:2b:f0:2c:32:dc:b9:b0:af:
        e6:55:44:f8:98:0b:76:02:72:40:a4:7a:1a:84:2a:40:e1:a8:
        e2:c2:f4:3a:8c:da:9b:83:ca:05:7b:81:b9:48:ac:6e:58:13:
```

kube api server 의 ssl/tls 인증서의 내용을 보면, 알 수 있는것들은 다음과 같다.

- `Issuer: CN = kubernetes` : 해당 인증서를 발급한 CN 은 kubernetes 이다.
- `Subject: CN = kube-apiserver` : 인증서의 주체의 CN은 kube-apiserver이다.
- `X509v3 Subject Alternative Name` : kube apiserver를 인증할 때 사용하는 별칭이다.
- `Validity` : 인증서가 얼마나 유효한지

### The kube-api server stopped again! Check it out. Inspect the kube-api server logs and identify the root cause and fix the issue.

kubeapi server 가 중지되었기 때문에, kubectl 을 이용한 로그는 확인할 수 없다.
따라서, container id 를 확인하기 위해, 런타임 환경에 따라 다르지만 docker ps -a 또는 crictl ps -a 등 명령어를 통해 container id를 가져온 후,
docker logs [container-id]를 통해 어떤 문제가 생겼는 지 살펴본다.

```bash
127.0.0.1:2379", }. Err: connection error: desc = "transport: authentication handshake failed: tls: failed to verify certificate: x509: certificate signed by unknown authority"
W0526 06:48:05.594458       1 logging.go:59] [core] [Channel #1 SubChannel #3] grpc: addrConn.createTransport failed to connect to {Addr: "127.0.0.1:2379", ServerName: "127.0.0.1:2379", }. Err: connection error: desc = "transport: authentication handshake failed: tls: failed to verify certificate: x509: certificate signed by unknown authority"
```

위와 같은 에러가 발생했고 kube-api server가 etcd 서버와 통신할 때 인증서에 문제가 있다는 것을 확인할 수 있다.
우선, kube-apiserver의 설정파일을 통해 ca file 이 어디있는 지 확인한다.

```bash
cat /etc/kubernetes/manifests/kube-apiserver.yaml | grep "etcd"
    - --etcd-cafile=/etc/kubernetes/pki/ca.crt
    - --etcd-certfile=/etc/kubernetes/pki/apiserver-etcd-client.crt
    - --etcd-keyfile=/etc/kubernetes/pki/apiserver-etcd-client.key
    - --etcd-servers=https://127.0.0.1:2379
```

위에서 etcd-cafile의 경로를 보면,` /etc/kubernetes/pki/ca.crt` 해당 경로는 클러스터 내에서 다양한 구성요소의 인증서를 발급하고 검증할 때 사용하는 루트 인증서 경로이다. 이 인증서는 새로운 노드나 사용자가 클러스터에 추가될 때, 이 CA를 이용해서 필요한 인증서를 발급하고 클러스터 서비스 내 상호 인증을 위해 이 CA가 인증기관으로서 작용된다.

그러므로, 해당 경로를 etcd 서버의 ca 인증서로 교체하여야 한다.

### certificate api

쿠버네티스 인프라 관리 팀에 새로운 멤버가 들어왔다. 해당 멤버가 쿠버네시스를 이용하기 위한 적절한 권한을 가진 인증서를 부여하려고 한다. 그 과정은 다음과 같다.

1. csr 파일 / key 파일 생성

```bash
# generate private key
openssl genrsa -out mykey.key 2048

# generate csr
openssl req -new -key mykey.key -out mycsr.csr -sujb "/CN=example.com/O=My organization/C=US"
```

2. 쿠버네티스에 CSR(certificate Signing Request) object 를 생성

```bash
# base64 로 인코딩
cat mycsr.csr | base64 -w 0

vi create-csr-object.yaml
# apiVersion: certificates.k8s.io/v1
# kind: CertificateSigningRequest
# metadata:
#   name: temp
# spec:
#   groups:
#   - system:authenticated
#   request: <위에서 인코딩한 값>
#   signerName: kubernetes.io/kube-apiserver-client
#   usages:
#   - client auth
```

3. 승인 또는 거절

```bash
kubectl certificate approve temp
kubectl certificate deny temp
```

### RBAC

RBAC 은 기관 내 개인의 유저가 클러스터에 접근할 때 규제를 둘 수 있게 해주는 것을 말한다. 권한을 가진 사람만 접근 가능하게 하려면, kube api server를 실행할 때, --authorization-mode 값에 `RBAC`을 포함시켜주면 된다.

```bash
kubectl describe role -n kube-system kube-proxy
Name:         kube-proxy
Labels:       <none>
Annotations:  <none>
PolicyRule:
  Resources   Non-Resource URLs  Resource Names  Verbs
  ---------   -----------------  --------------  -----
  configmaps  []                 [kube-proxy]    [get]

```

위 결과를 보면, kube-system namespace 에는 kube-proxy라는 role이 존재하고, 이 Role을 할당받은 유저는 kube-proxy라는 이름을 가진 리소스에 대해 get 요청만 할 수 있다는 것을 의미한다.
아래는, kube-proxy라는 role을 가진 주체를 뜻한다.

```bash
kubectl describe rolebindings.rbac.authorization.k8s.io -n kube-system kube-proxy
Name:         kube-proxy
Labels:       <none>
Annotations:  <none>
Role:
  Kind:  Role
  Name:  kube-proxy
Subjects:
  Kind   Name                                             Namespace
  ----   ----                                             ---------
  Group  system:bootstrappers:kubeadm:default-node-token
```

Role은 어떤 리소스에 대해 얼만큼의 접근을 허용할 지를 결정해주는 것을 의미하고, rolebinding 은 user/group/ 등 클러스터에 접근하는 주체에 role을 부여하는 것을 의미한다.
role/rolebinding 은 다음과 같이 생성할 수 있다.

```bash
kubectl create role developer --verb=list,create,delete --resource=pods
kubectl create rolebinding  dev-user-binding --role=developer --user=dev-user

# 추가로 여러개의 리소스에 대한 규칙을 하나의 role에 부여하는 것도 가능하다.
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  creationTimestamp: "2024-05-26T14:36:47Z"
  name: developer
  namespace: blue
  resourceVersion: "2165"
  uid: bd5331df-f342-4d58-88b5-fff760d5b765
rules:
- apiGroups:
  - ""
  resourceNames:
  - blue-app
  - dark-blue-app
  resources:
  - pods
  verbs:
  - get
  - watch
  - create
  - delete
- apiGroups:
  - apps
  resources:
  - deployments
  verbs:
  - create
```

Role 이 정상적으로 부여가 되었는 지 확인하기 위해선 아래와 같은 명령어를 작성하면 된다. 위 developer 에 명시된 role을 보면 pods에 대한 모든 접근권한, deployments에 대한 create권한만 생성된 것을 볼 수 있다.
따라서, deployments를 생성할 때는 성공적으로 수행되고, deployment 목록을 가져오는 api를 호출하게 되면 실패할 것이다.

```bash
kubectl create deployment test --image busybox --dry-run=client --as dev-user
deployment.apps/test created (dry run)

kubectl get deployments.apps --as dev-user
Error from server (Forbidden): deployments.apps is forbidden: User "dev-user" cannot list resource "deployments" in API group "apps" in the namespace "default"
```

### cluster role binding

앞서 role/rolebinding 과 cluster/cluster role binding 의 차이는 role은 특정 namespace내에서만 작용하는 반면, cluster role은 클러스터 전체에 걸쳐 작용한다.

### service account

service account는 pod 내에서 실행중인 어플리케이션이 쿠버네티스 api와 상호작용할 때 사용한다. 이 계정은 특정 네임스페이스에 속해있으며, pod level에서 쿠버네티스 리소스에 접근하는데 사용된다.
deployment에 service account 를 생성할 때는 spec.serviceAccountName 으로 service account 의 리소스 이름을 작성해주면 된다.

### Specifying imagePullSecrets on a Pod

private registry 에서 이미지를 pull하려고 할 때 사용하는 방법이다. 우선 docker config에 사용할 kubernetes secret 리소스를 생성해야 한다.

```bash
kubectl create secret docker-registry  <name> \
  --docker-server=DOCKER_REGISTRY_SERVER \
  --docker-username=DOCKER_USER \
  --docker-password=DOCKER_PASSWORD \
  --docker-email=DOCKER_EMAIL
```

secret 리소스를 생성하고 난 후 (literal 로 적어주는 것이 아닌 파일을 참조하는 것도 가능하다), private registry 에 접근권한을 주기 위해, pod의 설정파일에 `imagePullSecrets` 필드를 위에 만들어준 secret 이름으로 할당한다.

```
spec:
  imagePullSecrets:
   - name : <name>
  containers:
   ...
```

### security context

security context는 pod 또는 container 레벨에서 보안 수준을 정의하는데 사용된다. container 생성 시 command argument 에 특정 명령을 실행하는 주체가 누구인지 알고 싶다면 다음 명령어를 사용하면 된다.

```bash
kubectl exec [pod-name] -- whoami
```

```yaml
 apiVersion: v1
kind: Pod
metadata:
  name: multi-pod
spec:
  securityContext:
    runAsUser: 1001
  containers:
  -  image: ubuntu
     name: web
     command: ["sleep", "5000"]
     securityContext:
      runAsUser: 1002

  -  image: ubuntu
     name: sidecar
     command: ["sleep", "5000"]
```

위와 같이 specification 이 정의되어 있으면, pod level에서는 1001 로 작동한다. web container에서는 1002로 pod level의 security context를 덮으쓰게 된다. sidecar의 경우 pod level에서 정의한 1001의 context를 할당받게 된다.

```yaml
securityContext:
  capabilities:
    add: ["NET_ADMIN", "SYS_TIME"]
```

위 옵션은, 컨테이너가 실행될 때 컨테이너가 기본적으로 갖고 있지 않은 linux 기능을 가질 수 있다.
각 기능은 다음과 같다.

- NET_ADMIN : 네트워크 관련 수정 권한, ip,route table 변경이 가능
- SYS_TIME : 시간값 수정 권한

### network policy

aws의 security group과 비슷한 거다. 네트워크 트래픽을 제어하는 데 사용하는 쿠버네티스 리소스이다.

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: internal-policy
  namespace: default
spec:
  podSelector:
    matchLabels:
      name: internal
  policyTypes:
    - Egress
    - Ingress
  ingress:
    - {}
  egress:
    - to:
        - podSelector:
            matchLabels:
              name: mysql
      ports:
        - protocol: TCP
          port: 3306

    - to:
        - podSelector:
            matchLabels:
              name: payroll
      ports:
        - protocol: TCP
          port: 8080

    - ports:
        - port: 53
          protocol: UDP
        - port: 53
          protocol: TCP
```

위 specification 을 보면, `name=internal` label을 가진 pod에게 network policy 를 부여하는데, 외부 트래픽은 `name=mysql`을 가진 pod의 `3306` port 그리고, `name=payroll` label을 가진 pod의 `8080` port 로 나가는 트래픽만 허용한다.

추가로, kube-dns에서 dns resolve 를 하기 위해 TCP/UDP 53 번 트래픽을 허용한 것을 볼 수 있다.
