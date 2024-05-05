---
title: "[Network] 컨테이너 수준에서 네트워킹에 관하여"
subtitle: ""
date: 2024-05-04T17:26:01+09:00
lastmod: 2024-05-04T17:26:01+09:00
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

이전 [서버로 요청을 보낼 때 일어나는 일]({{% ref "/posts/network-basic.md" %}}) 에서 높은 수준에서 네트워크 트래픽이 어떻게 동작하는 지 알아봤었다.
이번에는 docker로 네트워크를 구축해보고 좀 더 낮은 커널 수준에서 네트워크를 구축하는 과정은 어떻게 이루어지는지 그 안에서 쓰이는 용어/ 명령어에 대해 알아보고자 한다.

### red 와 blue container 가 서로 트래픽을 주고 받을 수 있을까 ?

첫번째로 실습해볼 내용은, host pc 내에 red/blue container를 생성하고 2개의 network가 서로 통신할 수 있는지 테스트 해보는 것이다.

도커로는 이를 간단하게 구축할 수 있다. 도커 브릿지를 생성하고, 해당 네트워크에 컨테이너가 연결되도록 해주면 된다.

```bash
docker network create --driver bridge my_network

docker run -dit --name red --network mynet alpine
docker run -dit --name blue --network mynet alpine
```

위 과정은 너무 추상화되어 있다. 만약 컨테이너가 특정 네트워크에 할당되지 않고 생성되었을 경우, 두 컨테이너가 통신이 가능하도록 하려면 어떻게 해야되는 지 알아보자.
우선, 네트워크를 할당받지 않은 2개의 컨테이너를 생성하도록 한다.

```bash
docker network create --driver bridge my_network

docker run -dit --name red --network `none` alpine
docker run -dit --name blue --network `none` alpine
```

현재는 아래 그림과 같이, 트래픽을 서로 교환할 수 없는 상태이다. red 컨테이너가 다른 컨테이너와 통신을 하기 위해선 어떻게 해야될까?
여기서 등장하는 개념이 스위치이다. (docker network 환경에서는 switch와 bridge를 혼용해서 사용하곤 한다. 개념적으로는 다르다.)

<p align='center'>
<img src="/images/network/ping-error.png" width="100%" />
</p>

{{< admonition type=tip title="switch vs bridge" open=false >}}

1. 기본 개념과 목적
   브리지(Bridge): 브리지는 두 개 이상의 네트워크 세그먼트를 연결하여 하나의 통합된 네트워크처럼 작동하도록 돕습니다. 주로 네트워크 트래픽의 양을 줄이고, 서로 다른 네트워크 기술을 연결하는 데 사용됩니다.
   스위치(Switch): 스위치는 네트워크 내의 여러 장치들 사이에서 데이터 패킷을 전달하는 장치로, 각 장치에 대한 포트를 가지고 있어 효율적으로 데이터를 전송합니다. 스위치는 브리지의 기능을 포함하며, 더 많은 기능과 더 높은 효율성을 제공합니다.
2. 작동 방식
   브리지: 브리지는 MAC 주소를 사용하여 트래픽을 필터링하고, 필요한 경우에만 트래픽을 다른 세그먼트로 전달합니다. 이는 네트워크의 충돌 도메인을 줄이고 효율성을 높이는 데 도움이 됩니다.
   스위치: 스위치 역시 MAC 주소를 사용하지만, 각 연결된 장치를 위한 포트를 개별적으로 관리합니다. 이는 브리지보다 더 많은 동시 연결과 더 높은 트래픽 처리 능력을 가능하게 합니다. 스위치는 각 포트가 독립적인 충돌 도메인을 형성하므로 네트워크 성능이 크게 향상됩니다.
3. 용도 및 활용
   브리지: 작은 네트워크 또는 두 개의 분리된 네트워크 세그먼트를 연결하는 데 주로 사용됩니다. 브리지는 네트워크의 간단한 확장이 필요할 때 유용합니다.
   스위치: 현대의 네트워크 환경에서 스위치는 브리지보다 훨씬 널리 사용됩니다. 스위치는 그 자체로 많은 네트워크 세그먼트를 연결하고 효율적인 데이터 관리를 제공하여 대규모 네트워크 환경에서 중추적인 역할을 합니다.

{{< /admonition >}}

그럼 이번엔 스위치(브릿지)를 생성하고, 2개의 container가 서로 통신이 가능하도록 설정해보자.

#### 스위치 생성

```bash
ip link add name dockerbr0 type bridge
ip addr add 172.28.0.1/16 dev dockerbr0
ip link set dockerbr0 up
```

<br>
<p align='center'>
<img src="/images/network/bridge-interface-created.png" width="100%" />
</p>

위 명령어를 사용해서, dockerbr0라는 이름을 가진 `bridge`를 생성했고, 172.28.0.1/16 의 아이피 대역을 가지도록 했다.
<br>

<p align='center'>
<img src="/images/network/create-bridge.png" width="100%" />
</p>

현재는 위 그림과 같은 상태가 될 것이다. 여기서 red <-> blue container 간 통신이 가능하게 하려면 어떻게 해야할까 ?

red container에서 나가는 네트워크 인터페이스가 bridge의 네트워크 인터페이스와 연결이 되어야 하고, 추가로 blue container에서 나가는 네트워크 인터페이스가 bridge 네트워크 인터페이스가 연결이 되도록 구성해야 한다.
우선 red container에서 트래픽이 나가는 통로인 `veth-red`, blue container에서 트래픽이 나가는 `veth-blue`를 정의하고 각각의 인터페이스가 bridge 의 인터페이스인 `veth-red-br` / `veth-blue-br` 과 연결이 되도록 해보자.

#### veth pair 생성 그리고 브릿지 연결

```bash
ip link add veth-red type veth peer name veth-red-br
ip link add veth-blue type veth peer name veth-blue-br

# 브릿지 인터페이스로 등록
ip link set veth-red-br master dockerbr0
ip link set veth-red-br up

ip link set veth-blue-br master dockerbr0
ip link set veth-blue-br up

PID_RED=$(docker inspect --format '{{ .State.Pid }}' red)
sudo ip link set veth-red netns $PID_RED

nsenter --net=/proc/$PID_RED/ns/net -- ip addr add 172.28.0.2/16 dev veth-red
nsenter --net=/proc/$PID_RED/ns/net -- ip link set veth-red up

PID_BLUE=$(docker inspect --format '{{ .State.Pid }}' blue)
sudo ip link set veth-blue netns $PID_BLUE

nsenter --net=/proc/$PID_BLUE/ns/net -- ip addr add 172.28.0.3/16 dev veth-blue
nsenter --net=/proc/$PID_BLUE/ns/net -- ip link set veth-blue up
```

<br>
<p align='center'>
<img src="/images/network/veth-interface-created.png" width="100%" />
</p>

위와 같이 구성을 하게 되고 도식도를 그려보면 아래와 같다.
container red 는 veth-red(172.28.0.2)의 인터페이스를 통해 bridge 네트워크로 연결하고, container blue 는 veth-blue(172.28.0.3) 의 인터페이스를 통해 bridge의 네트워크로 연결했다.

<p align='center'>
<img src="/images/network/2container-network-connected.png" width="100%" />
</p>

#### 통신 테스트

위와 같이 구성을 하고 red에서 blue 로 핑을 날려보면 다음과 같이 성공적으로 통신이 가능해진 것을 볼 수 있다.

<p align='center'>
<img src="/images/network/ping-test.png" width="100%" />
</p>
