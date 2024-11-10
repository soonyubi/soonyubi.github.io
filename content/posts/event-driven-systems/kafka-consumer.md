---
title: "[Kafka] Consumer"
subtitle: ""
date: 2024-04-11T01:43:12+09:00
lastmod: 2024-04-11T01:43:12+09:00
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

## consumer 역할

컨슈머는 메시지를 가져오는 역할을 합니다. 이 때 어느 메시지를 가져와야 하는지를 결정해야 하므로 offset 을 관리하는 것이 중요합니다. 카프카에서는 consumer 가 관리하는 offset을 `__consumer_offset`토픽에 저장합니다. 기록하는 정보에는 컨슈머 그룹, topic, partition 정보가 함께 들어있습니다. 그 이유는 컨슈머는 언제나 컨슈머 그룹을 leave/join 할 수 있는데, 그 과정에서 새롭게 join 된 consumer가 메시지를 읽을 수 있어야 하기 때문입니다.

## 그룹 코디네이터

그룹 코디네이터의 역할은 컨슈머 그룹을 관리합니다. 주로 `리밸런싱` 과정을 통해 컨슈머 그룹내 컨슈머들을 관리합니다. 컨슈머들은 leave/join을 통해 그룹을 언제든지 나갔다 들어올 수 있고 이 과정은 코디네이터에게 알려져 컨슈머 그룹 내 컨슈머가 어떤 파티션에서 메시지를 읽어야 하는지 관리합니다. 또한 코디네이터는 컨슈머들의 변경을 감지하기 위해 `heartbeat`를 서로 주고 받습니다. `heartbeat`는 컨슈머가 살아있는지 잘 동작하는 지를 판단하고 컨슈머 그룹을 리밸런싱할지 말지 결정합니다.
다음은 heartbeat 관련 옵션입니다.

- heartbeat.interval.ms
  코디네이터에게 컨슈머는 자신이 살아있음을 나타내는 신호이다. 너무 큰 값을 가지게 되면 살아있는지 아닌지 판단을 빠르게 할 수없게 되고, 너무 작은 값을 가지게 되면 네트워크 call 이 많아지므로 적절한 값을 선택해야 한다. session.timeout.ms 의 1/3 수준으로 설정한다.
- session.timeout.ms
  컨슈머가 10초(default)동안 heartbeat를 받지 못하면, 컨슈머에게 문제가 생겼다고 판단하고 컨슈머 그룹내 컨슈머들을 리밸런싱합니다.
- max.poll.interval.ms
  컨슈머는 주기적으로 poll()을 요청해야 하는데, 5분(default)동안 요청을 하지 않게되면 코디네이터는 해당 컨슈머에 문제가 생겼다고 판단하고 그룹을 리밸런싱합니다.

## Static Membership

`리밸런싱`은 컨슈머가 나갔을 때 또는 그룹에 들어왔을 때 수행되며 cost가 높은 작업입니다.
예를 들어, 컨슈머가 10개로 구성된 컨슈머 그룹이 있고, 각 컨슈머의 hw/sw 업데이트 같은 작업이 있다고 가정할 때, 20번의 리밸런싱이 일어나게 되므로 이는 운영중인 서비스에서는 큰 타격을 입게 됩니다.

따라서, 카프카에서는 그룹 내 컨슈머들에게 id 를 부여하여, 기존 컨슈머 그룹의 일원이라는 것을 알리기 위함이라는 목적으로 static membership 이라는 개념을 도입합니다.

- static membership 적용하지 않았을 경우
<p align='center'>
<img src="/images/kafka/ch06/1.png" width="100%"/>
</p>

<p align='center'>
<img src="/images/kafka/ch06/2.png" width="100%"/>
</p>

<p align='center'>
<img src="/images/kafka/ch06/3.png" width="100%"/>
</p>

첫번째 이미지를 보면 /172.31.5.8 호스트는 0 /172.31.5.39 호스트는 1 /172.31.6.117 호스트는 2 번 파티션에 연결되어 있는 모습을 볼 수 있습니다. 여기서 /172.31.5.8 호스트를 종료시키고, 다시 서버를 키게 되면, 해당 컨슈머가 컨슈머그룹에 들어오면서 코디네이터는 리밸런싱을 진행합니다. 리밸런싱을 하게되면 다음과 같이 각각의 호스트가 연결되 파티션 번호가 바뀌는 모습을 볼 수 있습니다.

- static membership 적용했을 경우

<p align='center'>
<img src="/images/kafka/ch06/static_membership_1.png" width="100%"/>
</p>
<p align='center'>
<img src="/images/kafka/ch06/static_membership_2.png" width="100%"/>
</p>
<p align='center'>
<img src="/images/kafka/ch06/static_membership_3.png" width="100%"/>
</p>

첫번째 이미지를 보면 /172.31.5.39 호스트는 0 /172.31.5.8 호스트는 1 /172.31.6.117 호스트는 2 번 파티션에 연결되어 있는 모습을 볼 수 있습니다. 추가로 consumerID를 보면 위의 카프카에서 자체적으로 생성해준 id 와 다르게 고유의 id를 부여한 모습을 볼 수 있습니다. (`group.instance.id` property 에 값을 부여해주기만 하면 스태틱 멤버십을 사용할 수 있습니다.)

두번째 이미지는 8번 호스트가 비정상적으로 종료가 되었고, 코디네이터는 1번 파티션에 연결된 컨슈머의 종료로 리밸런싱을 시작합니다. 그 과정에서 39번 호스트가 8번 호스트가 담당하던 파티션을 담당하게 되었습니다. 추가적으로 볼 부분은, leave 명령을 통해서도 리밸런싱이 되면서 각각의 컨슈머들이 담당하던 파티션들이 변경되어야 하는데 스태틱 멤버십을 부여한 컨슈머들에게서는 바뀌지 않았다는 것에 주목하면 좋을 거 같습니다.
