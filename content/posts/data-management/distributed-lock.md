---
title: "[DATABASE] Control Concurrency in distributed system with Distributed Lock mechanism"
subtitle: ""
date: 2024-01-07T00:26:55+09:00
lastmod: 2024-01-07T00:26:55+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: ["data-management"]

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

## 개요

팔로우/언팔, 따닥으로 인한 중복데이터 생성, 아이유 티켓 결제하기 등 백엔드에선 동시성 문제나 같은 요청이 여러번 오는 경우가 많다. 이러한 케이스가 발생하지 않도록 로직 내에서 validation 을 잘하여 중복데이터가 발생하지 않도록 한다던지 DB 필드에 유니크 인덱스를 생성하여 중복데이터를 DB 쿼리단에서 막아준다던지로 해결할 수는 있지만, 서버 인스턴스가 여러대이고 많은 요청이 발생할 경우 이를 완벽하게 구현하기는 어렵다. 이러한 케이스를 서비스 로직 내에서 막기전에 그 앞단에서 락을 획득한 요청에 대해서만 operation 을 수행할 수 있도록 하는 방법을 설명하고자 한다.

각 언어마다 분산락을 구현하는 유명한 라이브러리는 다음과 같다. 하지만 다음 라이브러리마다 사용하는 분산락의 구현은 조금 차이가 있고, 우리 서비스에 현재로서 가장 적합한 분산락 알고리즘을 선택하는 것이 중요하다. 모든 trade-off 가 있기 때문이다. [라이브러리 모음](https://redis.io/docs/manual/patterns/distributed-locks)

## 분산락을 만족하는 속성

1. safety property : 1개의 client 만이 lock을 획득할 수 있어야하는 속성을 의미한다.
2. liveness property A : lock을 획득한 요청에서 에러가 발생하더라도 다른 요청이 lock을 획득할 수 있어야하는 속성을 의미한다.
3. liveness property B : 여러개의 redis instance를 운영하는 경우, 하나의 redis instance에서 장애가 발생하더라도, 나머지 인스턴스 사이에서의 분산락은 정상적으로 운영되어야 함을 의미한다. (`Fault tolerance`)

## 간단한 분산락 구현

1. 락을 획득한다.
2. 작업을 수행한다.
3. 락을 릴리즈한다.

분산락을 구현하는 간단한 과정으로 위와 같다.
먼저 락을 획득하는 구간에선, `SET lockKey my_random_value NX PX 30000` 명령어를 사용해서, lockKey 가 존재하지 않다면, 30000ms 동안 데이터를 저장해줘
이렇게 락을 획득하고 나서는, 동시성이 발생할만한 작업을 처리하고 다음과 같은 lua script 를 사용해서 락을 해제하거나 자신이 사용하고 있는 redis library api 를 이용해 lockKey 를 제거한다.

```typescript
@Injectable()
export class LockService {
  constructor(@InjectRedis() private readonly redisService: Redis) {}

  async acquireLock(lockName: string, timeout = 10000): Promise<boolean> {
    try {
      const result = await this.redisService.set(
        lockName,
        "OK",
        "PX",
        timeout,
        "NX"
      );
      return result === "OK";
    } catch (err) {
      console.log("acquireLock", err);
      throw err;
    }
  }

  async releaseLock(lockName: string): Promise<void> {
    try {
      const script = `
        if redis.call("get", KEYS[1]) == 'OK' then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
      `;
      const result = await this.redisService.eval(script, 1, lockName);
      console.log(result);
      return;
    } catch (err) {
      console.log("releaseLock", err);
      throw err;
    }
  }
}

// client.ts
if (!(await acquireLock("some-lockKey", 30000))) {
  throw LockAcquiredFailError();
}

// do something logic

await releaseLock("some-lockKey");
```

위의 경우, 락 획득에 실패하면 바로 에러를 핸들링하는 코드를 추가했지만, 스핀락 같은 방식을 사용해서 락을 획득할 때 까지 클라이언트가 무한정 시도하게 할 수 도 있다. 하지만 이러한 방식은 각각 장단점이 존재한다.
<br>만약 스핀락을 사용해서, 일정 갯수만큼 락 획득을 시도하다가 실패하는 코드를 추가하는 경우를 생각해보자.
로직을 수행하는 코드가 1초정도 걸리게 되고, 100개의 요청이 50ms 마다 lock을 획득하려 한다면, 2000번의 요청이 레디스 서버에 들어오게 된다. 요청의 갯수가 증가할수록 선형적으로 레디스의 부하는 커지게 된다. (pubsub 방식을 사용해 락을 획득하기를 기다리는 요청에 알림을 줌으로써 부하를 줄이는 방법도 있다.)
<br>반대의 경우에도 단점이 존재한다. 만약 사용자가 이벤트에 참여하려는 요청을 여러번 보낸다고 가정해보자. 사용자의 요청 중 하나만 성공하게 되고 나머지는 에러를 던지게 되므로 이 에러를 핸들링하는 클라이언트의 에러 핸들링 코드가 늘어나게 된다는 단점이 존재한다.

추가로 위 코드는 single instance 에 대한 분산락을 구현한 것이다. 만약 레디스 서버에 장애가 발생하게 된다면 해당 api 를 사용하는 모든 요청은 처리될 수 없게 되므로 Fault tolerance를 만족하지 못한다.

그렇다면, single instance에 대해서 위와 같이 exclusive lock 또는 barrier를 구현한다면 모든 중복 데이터 생성, 동시성 이슈를 해결할 수 있을까?? 그건 장담할 수 없다.
다음과 같은 상황을 고려해보자.<br><br>

<p align='center'>
<img src="/images/distributed-lock/duplicate-lock.png" width="80%"/>
</p>

위 케이스에서, 첫번째 요청에 대해 키가 없으면 lockKey에 대해 값을 세팅해줘 라고 레디스 서버에 요청을 보낸 바로 즉시 다음 요청이 같은 lockKey 에 대해 요청이 보내진 상태이다. safety property 를 만족하기 위해선, 첫번째 요청의 응답이 오기 전에 두 번째 요청은 락을 획득하는데 실패하여야 하지만, 아주 작은 타이밍에 같은 요청을 두 번 보내게 되고 어플리케이션 레벨에서는 동일한 2개의 락을 잡고 특정 로직을 수행하게 된다.
분산락에서 조심해야 할 점은, 어플리케이션 레벨에서 동시성을 막기 위해서 레디스를 이용해 분산락을 구현했지만, 레디스 분산락의 동시성까지 잡아줘야 한다는 관리 포인트가 하나 더 생기게 된다. <br>

또한, 레디스 클러스터를 운영중이라면 다음과 같은 문제가 생기게 된다. <br><br>

<p align='center'>
<img src="/images/distributed-lock/cluster-dlm.png" width="80%"/>
</p>

client 1이 락을 획득하다가, master node에서 장애가 발생해 slave node 가 master node 로 승격되는 상황을 가정해보자. 이 순간, 데이터의 복제는 지연이 발생하기 때문에 새로운 client2의 락 획득 요청에 대해 성공처리를 하게 되는 상황이 발생할 수 도 있다.

위 2가지 사례와 같은 락 획득에 대한 동시성을 잡아주기 위해 더 정교한 알고리즘인 `redlock` 알고리즘을 소개하고자 한다.

## multi instance 에서 분산락 구현 - redlock

분산환경에서 lock을 획득하기 위해서 클라이언트는 다음과 같은 일련의 작업들을 수행한다.

1. 락 획득을 시도하려고 하는 현재 시각(now)를 밀리초 단위로 가져온다.
2. 모든 redis에 순차적으로 락 획득을 요청한다.
3. 각 레디스에 락 획득에서의 걸린 시간을 구한다.
4. client 가 Redis 노드로부터 일정 수치 이상의 잠금을 획득하고 ((N+1)/2) , 락을 획득하기까지의 총 시간이 lock-key 유효시간보다 적은경우 락을 획득한 것으로 간주한다.
5. 만약 client 가 lock을 획득하지 못했다면 모든 인스턴스의 잠금을 해제한다.

<p align='center'>
<img src="/images/distributed-lock/redlock-1.png" width="80%"/>
</p>

<!-- annotation note -->

{{< admonition note "note" >}}
redlock 알고리즘은 각 클라이언트와 redis 서버간의 `시계가 동일하게 작동하는 것`을 가정한다.
{{< /admonition >}}

위와 같은 방식을 사용하면 무조건 safety and liveness property를 만족할 수 있는가? 라고 질문한다면 꼭 그렇지는 않다. 다음과 같은 상황을 생각해보자.
client A가 redis A,B,D 에게 락을 획득하고, client B는 redis C,E로 부터 락을 획득한 상황이다. 이 상황에서 redis D에 문제가 생겨 client A가 획득한 lock-key X가 휘발되었고, 장애가 복구되면서 client B가 redis D로 부터 lock-key Y 를 획득하였다. 이런 상황에선 client A, B 모두 락을 획득하였고 safety property가 깨지는 상황이다.

<p align='center'>
<img src="/images/distributed-lock/redlock-2.png" width="80%"/>
</p>

위와 같은 상황에서는, fsync=always option을 통해 데이터를 디스크에 쓸때마다 강제로 파일 시스템 캐시를 건너뛰고 바로 디스크에 동기화하도록 요구한다. 이렇게 함으로써 모든 쓰기 작업이 디스크에 완전히 동기화될 때까지 데이터베이스 작업이 반환되지 않도록 강제하여 락에 대한 safety 를 보장할 수 있다. 하지만 이런 방식을 사용하게 되면 파일 시스템 캐시를 사용하지 않기 때문에 쓰기작업의 대기 시간이 길어지고, 전체적인 성능이 저하도리 수 있다는 단점이 있다.

## 결론

서비스에 요구사항에 맞는 lock을 구현하는 것이 가장 중요하다. redlock 알고리즘이 분산락을 구현하는데에 있어서 가장 보편적이고 안전하다 ! 라는 관점으로 당장 우리 서비스에 적용시키기엔 관리 포인트가 더 늘어난다는 단점이 있고 그걸 반영함으로써 문제를 해결해야하는 구멍이 더 생겨나게 될 수도 있다. 가장 좋은 것은 어플리케이션 레벨에서 validation을 안전하게 잘하고 그걸로도 동시성이슈를 해결할 수 없다면 우리 서비스에 맞는 가장 좋은 분산락 구현을 하는 것이 좋다.
