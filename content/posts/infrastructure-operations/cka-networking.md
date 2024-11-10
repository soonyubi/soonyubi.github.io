---
title: "Cka Networking"
subtitle: ""
date: 2024-05-27T17:17:28+09:00
lastmod: 2024-05-27T17:17:28+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: ["kubernetes"]
categories: ["infrastructure-operations"]

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

### controlplane 노드에서 www.google.com 으로 ping 요청을 보낼 때

www.google.com 이 어떤 ip로 해석되고, 해석된 ip를 찾을 수 없을 경우 default route를 확인해서 트래픽을 해당 ip로 보내준다.
default gateway가 위치한 ip는 다음 명령어를 이용해서 확인할 수 있다.

```bash
ip route show default
```

### 어떤 cni를 사용하고 있는지 확인하는 방법

```bash
ls /etc/cni/net.d/ 로 설정파일을 확인할 수 있다.
```

### 2379 는 etcd 서버의 port이다.

### kubelet 의 container runtime endpoint 확인하기

```bash
ps aux | grep kubelet
```

### CNI binary file들이 위치한 경로

`/opt/cni/bin/`

### kubernetes 가 사용중인 cni plugin 확인하기

`/etc/cni/net.d/` 폴더 확인

### Identify the name of the bridge network/interface created by weave on each node.

ip link show 명령 후, weave라는 이름을 가진 네트워크 인터페이스를 확인할 수 있다.

### What is the POD IP address range configured by weave?

```bash
ip addr show weave
4: weave: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1376 qdisc noqueue state UP group default qlen 1000
    link/ether fe:79:46:c1:8f:18 brd ff:ff:ff:ff:ff:ff
    inet 10.244.0.1/16 brd 10.244.255.255 scope global weave
       valid_lft forever preferred_lft forever
```

### What is the range of IP addresses configured for PODs on this cluster?

1. 현재 cluster에 할당된 CNI 가 무엇인지 살펴본다.

ls /etc/cni/net.d --> weave

2. weave 네트워크를 사용중이므로, weave agent가 실행중인 pod를 살펴본다.

```bash
kubectl logs -n kube-system weave-net-sjlfk

INFO: 2024/05/27 22:22:26.605825 Command line options: map[conn-limit:200 datapath:datapath db-prefix:/weavedb/weave-net docker-api: expect-npc:true http-addr:127.0.0.1:6784 ipalloc-init:consensus=0 ipalloc-range:10.244.0.0/16 metrics-addr:0.0.0.0:6782 name:8e:7b:9a:f2:60:fb nickname:controlplane no-dns:true no-masq-local:true port:6783]
```

### What is the IP Range configured for the services within the cluster?

kube-api server 설정 파일을 확인해야 한다. `cat /etc/kubernetes/manifests/kube-apiserver.yaml`

### kube-proxy가 어떤 타입으로 실행중인지 확인하는 방법

```bash
kubectl logs kube-proxy -n kube-system

I0527 22:22:49.361543       1 server_others.go:72] "Using iptables proxy"
```
