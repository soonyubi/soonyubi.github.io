---
title: "[Kafka] Replication/ Controller/ Log"
subtitle: ""
date: 2024-04-08T11:56:11+09:00
lastmod: 2024-04-08T11:56:11+09:00
draft: false
author: "shy"
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: ["kafka"]

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

## 1. Kafka Replication

Kafka 는 데이터 파이프라인의 정중앙에 위치하는 메인 허브 역할을 한다. 만약 이 메인 허브에 문제가 생겨 전체 시스템을 운용할 수 없다면 어떻게 될까? 그러한 문제에 착안하여 안정성을 도모하기 위해 Kafka 내부에서 replication 을 구축하여 시스템 안정성을 높일 수 있다.

3대의 브로커가 운영중인 kafka 에 A 라는 토픽을 생성한 후, X 라는 메시지를 전송하게 되면 모든 서버는 X라는 메시지를 각각의 브로커들이 갖게 된다. replication 을 이용해서 같은 메시지를 여러 브로커가 같이 갖게 되어 안정성을 높일 수 있다.
(replication 이 될 때 topic 을 복제하는 것이 아닌 파티션을 복제하는 것임에 주의하자.)

### leader와 follower의 역할

- leader의 역할 1 : 메시지 읽고쓰기
  모든 메시지는 leader를 통해서만 읽고 쓰기가 가능하다. producer는 leader를 통해서만 메시지를 producing하고, consumer 는 leader 를 통해서만 메시지를 consume 한다.

- leader의 역할 2 : follower 감시하기

  leader 는 또한, follower 들이 메시지를 제대로 replication 하고 있는지 감시한다. leader 와 follower 는 `ISR`(InSyncReplica)라는 그룹에 묶여 있으며, follower 가 특정 주기안에 메시지를 복제하지 않는다면 leader는 해당 follower 를 그룹에서 방출시킨다. 이러한 과정을 하는 이유는 follower들은 언제든지 leader 로 승격될 수 있는 상태여야 하기 때문이다.

- leader의 역할 3 : 메시지 commit 하기

  모든 follower 들이 message 를 정상적으로 복제했다면, leader 는 해당 메시지를 commit 한다. 마지막 commit offset 위치를 `high watermark` 라고 한다. consumer는 commit 된 메시지만 읽을 수 있다. ( 만약 leader에 문제가 생겨 follower 들 중 하나가 leader 로 승격되는 케이스를 생각해보자. leader 로 부터 최신의 메시지를 복제하지 못한 상태에서 leader 로 승격된다면 메시지가 유실 되기 때문이다. )

### replication 은 어떻게 동작하는가

<p align='center'>
<img src="/images/kafka/ch04/_replication-flow.png" width="80%"/>
</p>

전통적인 메시지 큐 시스템인 RabbitMQ 에서는 follower 들이 메시지를 가져갈 때, 해당 메시지를 정상적으로 복제했다는 ACK 을 leader 로 전송함으로써 leader 가 복제가 되었음을 인지했다. Kafka에서는 ACK 요청을 제거함으로써 성능을 높일 수 있었다.

추가로, follower 들은 leader로 메시지를 pull 요청하는 방식을 채택하여 leader의 부하를 줄일 수 있었다. leader 가 push 하는 방식을 통해서 메시지를 전송하게 되면, 아직 메시지를 처리할 준비가 되지 않은 follower 들의 상태를 확인하기 위해서 네트워크 할당이 생기게 되므로 메시지를 replication 하는 과정에서 병목이 생기게 된다. 이를 줄이기 위해, 메시지를 처리할 준비가 된 follower 들이 leader 데이터를 fetch 함으로써 위에 설명한 병목을 줄이고 leader의 부하가 적어져 성능을 높일 수 있었다.

### leader epoch

leader epoch란 복구 과정에서 메시지 정합성을 위해 각 broker에게 어디까지 메시지가 정상적으로 commit 되었는지를 확인하기 위해서 사용한다.

다음과 같은 상황을 고려해보자

- High water mark 지연이 발생된 브로커가 leader 로 승격되는 경우

<p align='center'>
<img src="/images/kafka/ch04/leaderepoch-case1.png" width="80%"/>
</p>

1. 1번 offset 까지 복제를 마친 follower A 가 leader B 에게 fetchRequest(offset=2)
2. leader는 follower 가 1번 offset 까지 복제를 마쳤다고 판단 후, high water mark 를 상향조정한 후 fetchResponse 를 A에게 전송
3. A가 response 를 받지 못한 채 재시작하게 되면, high water mark 위 메시지들은 신뢰하지 못한다고 판단 후 파기
4. leader broker 가 다운되면, follower broker는 메세지2를 파기한 상태에서 leader 로 promotion -> 메시지 2 의 소실이 발생

- 모든 broker 가 다운되고, 불완전한 브로커가 leader 로 승격되는 경우

<p align='center'>
<img src="/images/kafka/ch04/leaderepoch-case2.png" width="80%"/>
</p>

1. follower, leader broker 모두 다운
2. follower 가 먼저 복구되어, leader 로 promotion
3. leader 가 된 A 가 새로운 메시지 m3를 받음
4. leader,follower 는 같은 high water mark를 가지고 있지만, 둘의 데이터는 다른 상황이 발생

위 두가지 사례를 통해, high water mark 만을 가지고 장애상황에 대한 복구를 했을경우, 데이터 정합성을 온전하게 맞출 수 없게 된다. 이를 해결하기 위해, `leader epoch` 라는 개념이 생겼고, leader epoch 는 새로운 leader 가 선출될 때 마다 1씩 증가한다.

첫번째 케이스에서는 다음과 같이 leader epoch를 사용한다.
재시작된 follower broker 는 leader epoch 에 요청을 보내고, offset 2를 응답받는다. 이 때, follower 는 high water mark 보다 높은 위치에 있는 메시지를 모두 삭제하지 않고, leader epoch로 응답받은 offset 까지의 메시지는 살려두는 방식으로 장애를 복구한다. 이를 통해 consumer는 소실없이 메시지를 소비할 수 있게 된다.

두번째 케이스에서는 다음과 같이 leader epoch를 사용한다.

<p align='center'>
<img src="/images/kafka/ch04/leaderepoch-3.png" width="80%"/>
</p>

1. 불완전하게 복제된 상태로 리더 브로커 A와 팔로워 브로커 B가 동시에 종료된다.
2. 팔로워 브로커 B가 먼저 재시작되어 새로운 리더로 선출됩니다.
3. 브로커 B는 리더이므로, LeaderEpoch 교환 과정을 거치지 않습니다.
4. 리더가 된 브로커 B가 프로듀서로부터 새로운 메시지 m3를 전송받습니다.
5. 브로커 A가 팔로워가 되어 재시작됩니다.
6. 리더 브로커 B에게 LeaderEpochRequest를 전송하고 1을 반환받습니다.
7. 현재 로컬에 가지고 있는 메시지 오프셋은 2이므로, 현재 리더의 start offset에 해당하는 1 8. 이후의 데이터는 모두 삭제합니다. 따라서, m2 메시지가 삭제됩니다.

## 컨트롤러

카프카의 컨트롤러는 리더 선출의 책임을 가지고 있다.
의도된 종료와 의도되지 않은 종료에서 의도된 종료가 다운타임을 최소화 시킬 수 있다.
그 이유는, 의도된 종료는 브로커가 종료되기 전 컨트롤러가 그룹내에서 리더를 선출하기 때문이다.

## 로그

로그는 segment파일에 저장된다.
로그는 1GB 가 넘게 되면, 롤링되어 새로운 파일에 저장한다.
로그를 관리하는 방법은 로그 삭제, 컴팩션이 있다.
