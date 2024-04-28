---
title: "K8s Secret"
subtitle: ""
date: 2024-04-27T20:50:57+09:00
lastmod: 2024-04-27T20:50:57+09:00
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

kube apiserver 는 쿠버네티스 클러스터의 접속할 수 있는 최초 방어선이다. 그래서 우리는 클러스터에 접속하는 것에 대해 2가지에 대한 결정을 해야한다.
누가 접근할 수 있고, 접근해서는 무얼 할 수 있는지

기본적으로 포드들 사이에선 서로 네트워크 통신을 할 수 있다. network policy를 세워두면, 포드들 간에 네트워크 통신에 대한 규칙을 생서할 수 있다.

## Authentication

kubernetes는 kube apiserver를 이용해서 service account 를 생성할 수 있다. 여기서 service account 는 외부 프로세스, 봇, 서드파티 모듈 같은 것들을 의미한다.
kubectl create serveraccount [service account name]

유저들의 대한 관리는 어떻게하는가? 여기서 유저라함은 쿠버네티스 관리자, 또는 개발자를 의미한다.
유저들은 kube apiserver 를 통해 쿠버네티스 클러스터에 요청을 할 수 있다. 이 때 특정 유저를 판별하기 위해 첫번재로 유저를 인증하고, 그 다음 요청을 처리한다.

어떻게 유저를 인증하는지?

1. static password file
2. static token file
3. certificates
4. ldap 같은 타사 인증 프로세스

첫번재로 static password file을 사용하기 위해선, kubeapi server를 설치할 때 basic-auth-file 옵션을 넘겨준다. 이때 basic-auth-file의 값은 csv파일인데, 패스워드, id, 유저id 로 구성되어 있다.
위 옵션을 지정하고 kubeapi server를 설치 또는 재설치하게 되면, 다음과 같이 kubeapi server로 유저를 인증할 수 있는 요청을 보낼 수 있다.

curl -v -k https://master-node-ip:6443/api/v1/pods -u "user1:12341234"

두 번째로 static token file 을 사용하기 위해선 token-auth-file 을 옵션으로 생성해주면 되고, 해당 값에는 토큰, 아이디, 유저 id 로 구성된 csv 파일을 넘겨준 후 kubeapi server 로 요청을 보낼 때는 다음과 같이 요청하면 된다.

curl -v -k https://master-node-ip:6443/api/v1/pods -h "Authorization: Bearer [token]"

위 방법은 추천되는 방법은 아니다. 해당 방법은 쿠버네티스 1.19 버전부터는 deprecated 되었다.

## TLS certificates

kube api server로 접근하는 client로는 cluster를 관리하는 admin이 될 수 있고, 포드를 어떻게 배치할 지 결정하는 scheduler 가 될 수 있고, controller manager, kube proxy 가 될 수 있다. 따라서 tls 통신을 하기 위해선 kube api server에 자체적인 인증서와 비밀키를 갖고 있어야 하고 client 도 각각 자신의 인증서와 비밀키를 갖고 있어야 한다.

kubeapi server로 들어온 요청은 etcd server 로 들어가야 하므로, 이 때는 kube api server가 etcd server 의 client 가 되어야하고 이때 tls 통신을 해야 한다면, etcd server / kubeapi server 모두 자신의 인증서와 비밀키를 갖고 있어야 한다.

추가로 인증서를 인증하기 위한 기관인 ca 가 필요하다. ca도 자신의 인증서와 비밀키를 갖고 있어야 한다.

인증서를 생성하기 위한 도구로는 easyrsa / openssl / cfssl 등이 있다.

1. generate key
   openssl genrsa -out admin.key 2048

   - openssl genrsa: RSA 개인키를 생성하는 명령어야.
   - out admin.key: 생성된 키를 admin.key라는 파일에 저장하라는 의미야.
   - 2048: 키의 크기를 지정하는 부분인데, 여기서는 2048비트를 사용했어. 보안 수준을 결정하는 중요한 부분이지.

2. certificate signing request
   openssl req -new -key admin.key -subj "/CN=kube-admin" -out admin.csr

   -openssl req: 인증서 요청이나 자체 서명 인증서(self-signed certificate) 생성을 위한 명령어야.
   -new: 새로운 CSR을 생성하겠다는 의미야.
   -key admin.key: CSR 생성 시 사용할 개인키 파일을 지정하는 거야. 여기서는 첫 번째 단계에서 생성한 admin.key를 사용해.
   -subj "/CN=kube-admin": CSR에 포함될 주체(subject)의 정보를 명시하는 옵션이야. 여기서 CN(Common Name)은 kube-admin으로 설정됐어.
   -out admin.csr: 생성된 CSR을 admin.csr이라는 파일에 저장하라는 의미야.

3. openssl x509 -req -in admin.csr -CA ca.crt -CAkey ca.key -out admin.crt
   -openssl x509: X.509 인증서 형식을 다루기 위한 명령어야. 여기서는 인증서를 생성하는데 사용돼.
   -req: CSR(admin.csr)을 사용해 인증서를 생성하겠다는 옵션이야.
   -in admin.csr: 입력 파일로 admin.csr을 사용하겠다는 의미야.
   -CA ca.crt: CA(Certificate Authority)의 인증서 파일을 지정하는 거야. 즉, 이 CA 인증서로 서명을 하겠다는 거지.
   -CAkey ca.key: 서명을 위해 사용될 CA의 개인키 파일을 지정하는 부분이야.
   -out admin.crt: 생성된 인증서를 admin.crt라는 파일에 저장하겠다는 의미야.

위 명령어는 admin 유저에 대한 인증을 위해 인증서를 발급하는 과정이다. 이 때 admin 유저가 인증서와 비밀키를 가지고 인증하려고 할 때 쿠버네티스는 어떻게 이 유저가 admin 유저인지 판별할 수 있을까?
그렇게 하기 우해선 csr을 생성할 때 , "/CN=kube-admin/O=system:master" 와 같이 master라는 그룹을 생성해서 서명 생성 요청을 발급하면 된다.

### certificate detail 확인하기

<p align='center'>
<img src="/images/k8s/cka/ch01/kubeapi-server-configuration.png" width="100%"/>
</p>

위 configuration 을 보면, etcd/ kublet 서버와 tls 통신을 하기 위한 인증서 비밀키가 저장된 공간을 확인할 수 있다.
추가로 클라이언트가 클러스터와 통신할 때 클라이언트의 인증서가 이 ca에 의해 인증된 인증서인지 확인하기 위한 목적의 인증서 파일의 위치를 확인할 수 있다.
추가로, kubeapi server 가 tls 통신을 하기 위해 사용하는 서버측 인증서/비밀키의 위치를 확인할 수 있다.

<p align='center'>
<img src="/images/k8s/cka/ch01/kubeapi-server-configuration.png" width="100%"/>
</p>

위 인증서를 통해 만료일/ issuer / 별칭 /Common Name 등등을 확인할 수 있다.

## certificate api

클러스터 관리자가 한명일 때 해당 관리자는 클러스터에 인증할 수 있도록, 인증서와 비밀키를 가지고 있다.
이때 새로운 클러스터 관리자가 추가되었을 때 해당 관리자는 개인의 비밀키와 인증서를 생성하고, 인증서에 대한 서명요청(csr)을 하기 위해 기존 관리자에게 인증서를 보내고
클러스터 내 ca 서버에서 인증을 받은 후 새로운 관리자에게 ca서버로 부터 인증받은 인증서를 건내주게 되면 새로운 관리자도 클러스터에 인증할 수 있게된다.

인증서에는 유효기간이 존재한다. 유효기간이 끝나게 되면, 새로운 관리자는 위 프로세스를 다시한번 거쳐서 ca서버로 부터 인증받은 인증서를 받아야 클러스터에 인증할 수 있게 된다.

그럼 ca는 어디에 존재하는가? ca는 인증서와 비밀키의 쌍이다. 보통은 마스터 노드에 위치한다.

위 과정을 계속 반복할 순 없으므로, certification api를 통해 새로운 인증서를 ca서버가 인증할 수 있도록 한다.

1. singing certificate request object 를 생성한다.
2. review request
3. approve request
4. share certs to users

처음에 유저는 key를 생성한다.
openssl genrsa -out jane.key 2048

그 다음 쿠버네티스 관리자에게 request 를 보낸다.
openssl req -new -key jane.key -subj="/CN=jane" -out jane.csr

관리자는 쿠버네티스 내에 signingCertificateRequest Object를 생성한다. (yaml 파일로)
여기에 request 필드에는 jane.csr 을 base64로 인코딩하여 집어넣는다.

쿠버네티스 클러스터에는 csr object 가 생성이되고, 기존 관리자는 해당 csr 을 다음명령어를 이용해 승인해준다.
kubectl certificate approve jane

위와 같은 인증서 관련 작업은 마스터 노드의 controller manager 에 의해 수행된다.
인증서에 인증을 하기 위해서는 루트 인증서와 비밀키가 필요하다. 따라서, controller manager 서버를 까보면
--cluster-signing-cert-file / cluster-signing-key-file 옵션이 들어있는것을 확인할 수 있다.
