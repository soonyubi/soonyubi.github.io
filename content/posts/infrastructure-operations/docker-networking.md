---
title: "[Network] docker network 구성"
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
categories: ["infrastructure-operations"]

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

### 서로 다른 LAN에 속한 컨테이너가 통신을 하기 위해서

<p align='center'>
<img src="/images/network/network_with_router.png" width="100%" />
</p>

앞선 예제와 비슷하게, green, orange container를 생성해서 dockerbr1 브릿지에 연결시켜놓은 상태이다.
여기서 dockerbr0에 속한 red container에서 dockerbr1에 속한 green container에 요청을 보내고 싶으면 어떻게 해야 할까?

방법은 여러가지가 있을 수 있다.

1. routing table
2. 가상의 인터페이스를 사용해서 브릿지 연결
3. ip tunneling
4. 특정 bridge에 모든 컨테이너 연결하기

이번 실습에서 해볼 것은, 위 그림과 같이 router를 통해 서로 다른 네트워크에 속한 컨테이너가 통신이 가능하도록 하려고 한다.

#### 라우팅 컨테이너 생성 후 브릿지에 연결

```bash
docker run -dit --name router --network network1 --privileged alpine
docker network connect network1 router

# 라우터 컨테이너에 IP 포워딩 설정
docker exec router sh -c "echo 1 > /proc/sys/net/ipv4/ip_forward"
```

router 역할을 수행하는 컨테이너를 생성하고, 라우터로 들어오는 트래픽을 다른 네트워크로 전달할 수 있도록 IP 포워딩 설정을 해주었다.

#### 라우팅 규칙 설정

```bash
# NAT 설정 (옵션 - 외부 네트워크로의 접근 필요 시)
docker exec router iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
docker exec router iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE

```

NAT 설정을 하여, 외부 네트워크와 통신할 수 있도록 구성했다.
NAT 설정을 하는 이유는 컨테이너가 다른 네트워크(예: 인터넷 또는 다른 사설 네트워크)와 통신할 때 원본 IP 주소를 그대로 사용하면, 목적지 네트워크에서 해당 IP 주소를 라우팅할 수 없거나 보안 정책에 의해 차단될 수 있다. MASQUERADE는 이런 문제를 해결하고, 외부 네트워크에서 응답을 해당 컨테이너로 정확히 라우팅할 수 있게 한다. 또한 원본 IP 주소가 변경되므로, DHCP 서버에 의해 IP를 동적으로 할당받는 환경에서 유용하게 사용된다.

#### 각 컨테이너에서 라우터를 통해 다른 네트워크로 라우팅이 가능하도록 게이트웨이 설정

```bash
# 예시로, 컨테이너에서 수동으로 라우팅 설정
docker exec container1 route add default gw [router의 network1 IP]
docker exec container2 route add default gw [router의 network2 IP]

# 내부적으로는 아래와 같은 동작을 한다.
# ip route addr [network2 subnet] via  [router의 network1 interface]
# ip route addr [network1 subnet] via  [router의 network2 interface]
```

위 명령어는, 특정 container에서 외부의 ip로 요청을 보냈을 때, routing table에 정의되있지 않을 경우, 해당 트래픽을 router로 이동하라고 routing table에 추가하는 것을 의미한다.
여기서 router 의 ip는 다음 명령어를 통해 알 수 있다.

```bash
docker inspect -f '{{json .NetworkSettings.Networks}}' router
```

```json
{
  "my_network": {
    "IPAMConfig": null,
    "Links": null,
    "Aliases": null,
    "MacAddress": "02:42:ac:13:00:04",
    "NetworkID": "84436b4c6c98e7070092e8c8ad0cdeb90496fbdc7f6d55917dce7ed1bf952bf2",
    "EndpointID": "3d4a37627d5f87049c8598387b4a2f83937ba74feb91f66de0ac783756feda5d",
    "Gateway": "172.19.0.1",
    "IPAddress": "172.19.0.4",
    "IPPrefixLen": 16,
    "IPv6Gateway": "",
    "GlobalIPv6Address": "",
    "GlobalIPv6PrefixLen": 0,
    "DriverOpts": null,
    "DNSNames": ["router", "339a3536c1b8"]
  },
  "my_network2": {
    "IPAMConfig": {},
    "Links": null,
    "Aliases": [],
    "MacAddress": "02:42:ac:14:00:04",
    "NetworkID": "cb84797946682d27e81bf967ee4a225f86462cbe23b9054f47cc18a35316d2ba",
    "EndpointID": "65b88bb2e2cef144fce7282a6e34b13841a7866eca6a6d9b86453dc2aaf404f6",
    "Gateway": "172.20.0.1",
    "IPAddress": "172.20.0.4",
    "IPPrefixLen": 16,
    "IPv6Gateway": "",
    "GlobalIPv6Address": "",
    "GlobalIPv6PrefixLen": 0,
    "DriverOpts": {},
    "DNSNames": ["router", "339a3536c1b8"]
  }
}
```

routing table 정보는 다음과 같이 확인할 수 있다.

```bash
[host] $ docker exec red ip route show
default via 172.19.0.4 dev eth0
default via 172.19.0.1 dev eth0
172.19.0.0/16 dev eth0 scope link  src 172.19.0.2

[host] $ docker exec blue ip route show
default via 172.19.0.4 dev eth0
default via 172.19.0.1 dev eth0
172.19.0.0/16 dev eth0 scope link  src 172.19.0.3

[host] $ docker exec green ip route show
default via 172.20.0.4 dev eth0
default via 172.20.0.1 dev eth0
172.20.0.0/16 dev eth0 scope link  src 172.20.0.2

[host] $ docker exec orange ip route show
default via 172.20.0.4 dev eth0
default via 172.20.0.1 dev eth0
172.20.0.0/16 dev eth0 scope link  src 172.20.0.3

[host] $ docker exec router ip route show
default via 172.19.0.1 dev eth0
172.19.0.0/16 dev eth0 scope link  src 172.19.0.4
172.20.0.0/16 dev eth1 scope link  src 172.20.0.4
```

위와 같이 구성을 하고 나면, 서로 다른 LAN 에 위치한 컨테이너들이 서로 통신을 할 수 있게 된다.

### DNS :

다음으로 해볼 것은 dns 서버를 구성하여 `docker exec red ping green` 처럼 green ip를 통해 통신을 하는 것이 아닌, domain 이름을 사용해서 통신을 하려고 한다.
dns의 대한 설명은 [여기](https://aws.amazon.com/ko/route53/what-is-dns/) 를 참고한다.

<p align='center'>
<img src="/images/network/dns.png" width="100%" />
</p>

red -> green 으로 요청을 보낼 때, red 는 green이란 도메인을 ip로 바꾸기 위해 /etc/hosts 파일을 참조하여 ip로 변환하는 작업을 수행할 것이다.
하지만, red는 green을 해석할 수 없어, ping 요청은 실패할 것이다.

red container의 /etc/hosts 파일에 green 172.29.0.2 를 적고, ping 요청을 하면 성공적으로 응답이 떨어지는 것을 확인할 수 있다.
하지만, 우리가 통신해야 되는 ip 들이 많아지게 되면, 관리하기가 힘들기 때문에 이를 중앙에서 제어해주는 서버를 dns server라고 한다.

dns server의 역할은 도메인을 ip로 변환해주거나, ip를 도메인으로 변환(PTR), 캐싱, 부하 분산, 보안, 도메인관리등이 있다.
도메인을 ip로 변환해주는 간단한 dns server를 생성하는 실습을 해보고자 한다.

```conf
# dnsmasq.conf 정의
# dnsmasq.conf
interface=eth0
domain-needed
bogus-priv
expand-hosts

# 컨테이너 이름에 대한 정적 DNS 매핑
address=/red/172.28.0.2
address=/blue/172.28.0.3
address=/green/172.29.0.2
address=/orange/172.29.0.3
```

```bash
# dns server 생성
docker run -d --name dns-server --network bridge1 --cap-add=NET_ADMIN andyshinn/dnsmasq
docker network connect bridge2 dns-server

docker cp dnsmasq.conf dns-server:/etc/dnsmasq.conf
docker restart dns-server

dns_server_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' dns-server)
echo $dns_server_ip

docker run -dit --name red --network bridge1 --ip 172.28.0.2 --dns $dns_server_ip alpine ash
docker run -dit --name blue --network bridge1 --ip 172.28.0.3 --dns $dns_server_ip alpine ash
docker run -dit --name green --network bridge2 --ip 172.29.0.2 --dns $dns_server_ip alpine ash
docker run -dit --name orange --network bridge2 --ip 172.29.0.3 --dns $dns_server_ip alpine ash
```

dns 서버 설정 파일에 각각의 호스트가 어떤 ip를 가지고 있는지 정의하고 dns-server를 실행시켜준 후
기존에 운영중인 red,green,blue,orange 에 dns_server ip를 적용시키기 위해 모두 중지하고 재실행 시켜줬다.

위와 같이 구성을 하게 되면, 서로 다른 LAN 에 속해있는 red->green, green->blue .. 로 요청을 보낼 때 ip를 쓰지않고 도메인 이름을 사용하여 보내는 것이 가능하다.
`docker exec red traceroute green`명령을 호출할 때, 정말로 dns server 에서 응답을 받는지 확인하기 위해, `tcpdump` 패키지를 사용하여, dns 쿼리를 캡쳐해보자.

```bash
docker exec -it red apk add --no-cache tcpdump
docker exec -it red tcpdump -i any -n 'udp port 53'
docker exec red traceroute green
```

<br>
위와 같이 53번 port로 리스닝하고 있는 서버를 그대로 두고, 새로운 탭을 열어 `docker exec red traceroute green` 하게 되면 아래와 같이 dns server로 부터 적절히 응답을 받는 것을 볼 수 있다.

<br>
<p align='center'>
<img src="/images/network/dns-query.png" width="100%" />
</p>

{{< admonition type=note title="" open=false >}}

- AAAA 및 A 쿼리: red 컨테이너가 green의 IP 주소를 얻기 위해 DNS 쿼리를 보냄.
- 응답: DNS 서버가 IPv4 주소에 대해서만 응답을 제공하고, IPv6 주소에 대해서는 응답하지 않음.
- PTR 쿼리: IP 주소로부터 호스트 이름을 얻기 위한 요청이 있었으나, 적절한 응답을 받지 못함 (NXDomain).
- 로컬호스트 DNS 캐시: 127.0.0.11는 도커 내부적으로 사용하는 로컬 DNS 캐시 주소로, 컨테이너 내부의 DNS 요청 처리를 담당.

{{< /admonition >}}
