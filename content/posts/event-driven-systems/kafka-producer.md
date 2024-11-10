---
title: "[Kafka] Producer"
subtitle: ""
date: 2024-04-10T17:47:50+09:00
lastmod: 2024-04-10T17:47:50+09:00
draft: false
author: ""
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

## Partitioner

<p align='center'>
<img src="/images/kafka/ch05/partitioner.png" width="80%"/>
</p>

카프카는 메세지를 병렬로 효율적으로 처리하기 위해 여러개의 파티션으로 구성할 수 있다. 메시지는 파티션으로 보내지고 파티션 내 로그 세그먼트에 저장되어 카프카로 전송된다. 여러개의 파티션이 구성되어 있을 경우 어느 파티션으로 보내줘야 하는 지 결정하는 역할을 하는 것이 파티셔너이다.

파티셔너는 메시지의 키를 해시하여 어떤 토픽의 어떤 파티션으로 보내줘야 할 지 매핑하는 테이블을 관리한다. 파티션의 갯수가 변경되면, 해시테이블도 변경되므로 파티션의 갯수를 늘릴 때는 유의해야한다.

메시지의 키값은 필수값이 아니므로, null 일 경우 카프카는 자체적으로 해당 메시지를 어느 파티션으로 보낼 지 결정한다. 다음은 그 방법이다.

1. Round Robin

여러개의 파티션으로 구성되어 있는 경우, 파티셔너는 메시지를 랜덤으로 파티션에 메시지를 전송한다. 전송된 메시지는 I/O interrupt 를 줄이기 위해, 파티션의 버퍼 메모리 영역에서 대기를 하는데, 라운드 로빈을 사용할 경우, 버퍼 메모리가 차기 전까지 메시지가 대기할 수 있기 때문에 효율적이지 못하다. ( 버퍼 메모리 영역 내에서 얼만큼 기다리는 지 결정할 수 있는 설정값(linger.ms)가 있기는 하다. )

2. Sticky Partitioning

라운드로빈의 비효율적인 전송을 방지하기 위해, 버퍼 메모리 영역 내 가장 빠르게 배출될 수 있을만한 파티션을 찾아 파티셔너는 해당 파티션으로 메시지를 전송한다. 버퍼 메모리가 더 빨리 채워질 수 있기 때문에 라운드로빈보다 효율적으로 메시지를 처리할 수 있는 장점이 있다.

하지만, 이러한 방식은 메시지의 전송 순서가 보장이 되지 않기 때문에 메시지의 순서가 중요하지 않은 경우일 경우 sticky partitioning 전략을 사용하면 좋다.

## Batch

카프카는 토픽의 처리량을 높이기 위해, 메시지를 하나하나 처리하는 것이 아닌 모았다가 한번에 처리한다. 이렇게 한번에 처리하는 이유는 I/O interrupt 를 줄이기 위해서다.

배치와 관련된 설정값은 다음과 같다.

1. buffer.memory
   기본값은 32MB 이고, buffer.memory > batch.size \* #partition 이어야 함을 주의하자.
2. batch.size
   파티션내 메시지들이 배치 전송을 위해 묶는 단위이다. 기본값은 16 KB 이고, 16KB 가 되면은 파티션내 메시지들을 한번에 처리한다.
3. linger.ms
   메시지가 버퍼 메모리에서 대기하는 시간이다. 기본값은 0ms이다.

카프카를 사용할 때 주의할 점은 처리량을 높일 것인지 지연없는 전송을 할 것인지 결정해야 한다. 운영중인 어플리케이션에 따라 위 설정값들을 조절하며 어느것이 어플리케이션에 최적화된 값인지를 찾아야 한다.

지연없는 전송을 위해선, batch.size / linger.ms 값을 줄이면 되고, 처리량을 높이기 위해선 늘리면 된다.

처리량을 높이기 위해서, 카프카는 여러가지 압축포맷을 지원하기도 한다.

## 전송방식

### 적어도 한번 전송

메시지를 프로듀서가 보내고 그에 대한 ACK 이 오지 않는다면, 같은 메시지를 한번 더 보내는 방식이다. 카프카는 기본적으로 적어도 한번 전송으로 동작하고 이는 메시지가 중복 가능성이 있다는 것을 의미한다.

### 최대 한번 전송

메시지를 프로듀서가 보내고 ACK을 기다리지 않고 다음 메시지를 보낸다. 메시지의 유실이 있을 수 있기 때문에 log 나 IoT 같은 환경에서 사용된다.

### 중복없는 전송

프로듀서는 메시지를 보낼 때 헤더에 PID / Sequence Number 를 포함하여 브로커에게 전달한다. 브로커는 위 두 값을 비교하여 처리된 메시지인지 아닌지를 판단하고, 만약 처리된 메시지라면 메시지를 저장하지 않고 ACK 만 응답한다.

PID / Sequence Number 는 브로커의 메모리 뿐 아니라, replication log 에도 작성되어 브로커가 다운되어 follower 가 promotion 될 때도 중복 없는 메시지 전송이 가능하다.

여기서 주의해야 할 점은 중복없는 전송은 정확히 한번 전송과 구별된다는 점이다.

중복없는 전송을 하기 위해선 프로듀서에 다음과 같은 설정값들이 필요하다.

- enable.idempotence = true
  기본값은 false 이며, 이 값이 true 로 설정되면, 다음 3가지의 옵션을 수정해야 한다.
- max.in.flight.requests.per.connection = 1~5
  브로커로부터 ack 이 오지 않았을 때 connection 에서 보낼 수 있는 최대 요청 수
- acks = all (-1)
  프로듀서는 모든 리더 브로커와 ISR 에 포함된 브로커들이 모두 메시지를 기록했다는 요청을 기다린다. 이렇게 됨으로써 메시지를 중복없이 처리할 수 있게 된다.
- retries > 0
  ack 을 받지 못한 경우 몇 번 재시도를 해줘야 하는지 결정해주는 값이다.

<p align='center'>
<img src="/images/kafka/ch05/snapshot_log.png" width="80%"/>
</p>

위와 같이 `ProducerId` / `lastSequence` / `firstSequence` 가 기록된 모습을 볼 수 있다.

### 정확히 한번 전송

<p align='center'>
<img src="/images/kafka/ch05/kafka-transaction.png" width="80%"/>
</p>

카프카는 정확히 한번 전송을 수행하기 위해, 별도의 프로세스가 존재하는데 이를 트랜잭션 API라고 한다.

카프카에서 트랜잭션을 관리하기 위해 별도의 브로커에는 `transaction coordinator` 가 존재한다.

카프카 트랜잭션을 사용하기 위해선, 중복없는 전송에서 설정했던 값 이외에 `transactional_id_config` 값이 추가되어야 한다. 이 때 프로듀서마다 서로 다른 값으로 해당 설정값을 구성해야 한다.

1. 트랜잭션 코디네이터 찾기
   프로듀서는 브로커에게 `FindCoordinatorRequest` 요청을 보내 코디네이터를 찾게 된다. 이 과정에서 트랜잭션 코디네이터는 PID(producer id) 와 TID(transaction id) 를 매핑하고 트랜잭션을 관리한다.

2. initTransaction()
   프로듀서는 위 메서드를 통해 초기화를 하며, 이 때 InitPidRequest 요청을 트랜잭션 코디네이터로 보낸다. 트랜잭션 코디네이터는 이 과정에서 PID / TID 를 매핑하고 PID Epoch를 한 단계 올린다. (epoch를 증가시키는 이유는, 메시지 안정성을 높이기 위함이다.)

   ```bash
   peter-transaction-01::TransactionMetadata(transactionalId=peter-transaction-01, producerId=3000, producerEpoch=0, txnTimeoutMs=60000, state=Empty, pendingState=None, topicPartitions=Set(), txnStartTimestamp=-1, txnLastUpdateTimestamp=1712735064600)
   ```

   트랜잭션 초기화에 해당하는 로그이다. PID 는 3000 / state=Empty / topicPartition은 empty set 이다.

3. beginTransaction()
   이 과정에서 프로듀서는 토픽 파티션 정보를 트랜잭션 코디네이터에 전달하고 트랜잭션 코디네이터는 트랜잭션 로그에 TID, 파티션 정보를 기록하고 트랜잭션 상태를 `ongoing`으로 표시한다. 추가로 트랜잭션 코디네이터는 해당 트랜잭션에 대해 타이머를 설정하고 1분을 넘길 경우 해당 트랜잭션을 실패로 간주한다.

   ```bash
   peter-transaction-01::TransactionMetadata(transactionalId=peter-transaction-01, producerId=3000, producerEpoch=0, txnTimeoutMs=60000, state=Ongoing, pendingState=None, topicPartitions=Set(peter-test05-0), txnStartTimestamp=1712735065999, txnLastUpdateTimestamp=1712735065999)
   ```

   state=ongoing으로 변경되었고, topicPartitions=Set(peter-test05-0) 에서 트랜잭션이 시작했음을 알 수 있다.

4. 메시지 전송

   프로듀서는 메시지를 전송하면서, PID / epoch / sequence number 를 메시지에 담아 전송한다.

   ```
   | offset: 0 CreateTime: 1712735065962 keysize: -1 valuesize: 52 sequence: 0 headerKeys: [] payload: Apache Kafka is a distributed streaming platform - 0
   baseOffset: 1 lastOffset: 1 count: 1 baseSequence: -1 lastSequence: -1 producerId: 3000 producerEpoch: 0 partitionLeaderEpoch: 0 isTransactional: true isControl: true position: 120 CreateTime: 1712735066161 size: 78 magic: 2 compresscodec: NONE crc: 1514605644 isvalid: true

   | offset: 1 CreateTime: 1712735066161 keysize: 4 valuesize: 6 sequence: -1 headerKeys: [] endTxnMarker: COMMIT coordinatorEpoch: 0
   ```

   lastSequence: 0 producerId: 3000 producerEpoch: 0 값을 통해 메시지에 PID / epoch / sequence number 가 담겨 전송된다는 것을 확인할 수 있다.

   offset 1번에 담긴 로그를 보면 기존 로그와 형태가 조금 다른 걸 볼 수 있는데, 이 메시지는 `control message` 이다. 컨슈머에 read_committed option이 활성화 되어 있다면 이 메시지 앞에 있는 메시지들만 읽을 수 있다.

5. 메시지 전송 완료

메시지를 전송완료하게 되면, 프로듀서는 commitTransaction() 또는 abortTransaction() 메서드를 호출해야만 한다. 커밋이 되었다면 트랜잭션 코디네이터는 해당 트랜잭션에 대한 로그인 prepareCommit 또는 prepareAbort를 로그에 기록한다.
