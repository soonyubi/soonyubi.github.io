---
title: "Mysql hash join"
subtitle: ""
date: 2023-10-04T23:19:40+09:00
lastmod: 2023-10-04T23:19:40+09:00
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

## hash join

mysql에서 join을 실행하는 유일한 알고리즘은 BNL 이었다. BNL은 외부 테이블을 스캔하면서 일정 크기의 블록(메모리 버퍼)를 채울 때까지 행을 읽고, 내부 테이블을 스캔하면서 외부 테이블의 메모리 버퍼에 있는 행과 일치하는 지 확인하고 결합한 후 결과세트에 추가하는 방식이다.

그 이후로는, hash join이라는 BNL 보다 더 효율적으로 쿼리를 실행하게 된다. hash join은 두 입력 사이에서 일치하는 행을 찾기 위해 해쉬 테이블을 이용하는 방법이다.

```mysql
SELECT
  given_name, country_name
FROM
  persons JOIN countries ON persons.country_id = countries.country_id;
```

### build phase

위와 같은 쿼리가 있다고 가정해보자. build phase 에서 in-memory hash table의 key를 만들기 위해 두 테이블 중 작은 것(바이트 수가)을 선택해 join 조건으로 걸린 column을 해쉬 테이블의 key로 사용하고 해쉬테이블의 모든 행을 저장한다.

<p align='center'>
<img src="/images/mysql/build-phase-1.jpg" width="80%"/>
</p>

### probe phase

이제 나머지 남은 persons 테이블의 country_id를 해쉬 테이블의 key와 매칭해서 상수시간에 데이터를 조회하여 client에 넘겨주게 된다.

위와 같은 작업이 정해진 메모리 내(`join_buffer_size`)에서 이루어 질 수 있다면, hash join은 매우 잘 작동한다. 하지만 빌드 입력이 메모리 제한보다 크다면 디스크로 넘치게 된다.

<p align='center'>
<img src="/images/mysql/probe-phase-1.jpg" width="80%"/>
</p>

### spill to disk

디스크 기반의 작업으로 넘치게 되면 성능에 부정적인 영향을 끼친다. 오로직 디스크에 쓰게 되면 더 나쁜 성능저하를 초래하게 되므로, chunk 단위로 잘라서 디스크에 입력한다. 이 때도 해쉬 함수가 사용되어 최악의 성능저하를 막는다.

이렇게 넘치는 데이터는 chunk에 쓰이는 이유는 dist IO를 최대한 피하기 위해서이다. 추가로, probe phase에서 해쉬 테이블에서 일치하는 모든 행을 찾는 것 뿐 아니라 디스크에 쓰여진 행에서도 해쉬함수를 이용해 일치하는 행을 찾는다.

<p align='center'>
<img src="/images/mysql/probe-phase-on-disk.jpg" width="80%"/>
</p>

## hash join은 언제 사용할 수 있는지 ?

mysql 8.0.18 이전에서는 hash join을 사용하기 위해선, `optimizer_switch` flag를 건드려서 hash join 을 사용하도록 해야했다. 그 이후 버전에서는 기본적으로 equi-join에 대해 hash join을 사용한다. 꼭 equi-join이 아니더라도(cartesian이라도) hash join을 사용한다.

다음은 hash join을 사용하는 예제이다.

```mysql
# inner non-equi join

mysql> EXPLAIN FORMAT=TREE SELECT * FROM t1 JOIN t2 ON t1.c1 < t2.c1\G
*************************** 1. row ***************************
EXPLAIN: -> Filter: (t1.c1 < t2.c1)  (cost=4.70 rows=12)
    -> Inner hash join (no condition)  (cost=4.70 rows=12)
        -> Table scan on t2  (cost=0.08 rows=6)
        -> Hash
            -> Table scan on t1  (cost=0.85 rows=6)

# Semijoin:

mysql> EXPLAIN FORMAT=TREE SELECT * FROM t1
    ->     WHERE t1.c1 IN (SELECT t2.c2 FROM t2)\G
*************************** 1. row ***************************
EXPLAIN: -> Hash semijoin (t2.c2 = t1.c1)  (cost=0.70 rows=1)
    -> Table scan on t1  (cost=0.35 rows=1)
    -> Hash
        -> Table scan on t2  (cost=0.35 rows=1)

# Antijoin

mysql> EXPLAIN FORMAT=TREE SELECT * FROM t2
    ->     WHERE NOT EXISTS (SELECT * FROM t1 WHERE t1.c1 = t2.c1)\G
*************************** 1. row ***************************
EXPLAIN: -> Hash antijoin (t1.c1 = t2.c1)  (cost=0.70 rows=1)
    -> Table scan on t2  (cost=0.35 rows=1)
    -> Hash
        -> Table scan on t1  (cost=0.35 rows=1)

1 row in set, 1 warning (0.00 sec)

mysql> SHOW WARNINGS\G
*************************** 1. row ***************************
  Level: Note
   Code: 1276
Message: Field or reference 't3.t2.c1' of SELECT #2 was resolved in SELECT #1

# Left outer join:
mysql> EXPLAIN FORMAT=TREE SELECT * FROM t1 LEFT JOIN t2 ON t1.c1 = t2.c1\G
*************************** 1. row ***************************
EXPLAIN: -> Left hash join (t2.c1 = t1.c1)  (cost=0.70 rows=1)
    -> Table scan on t1  (cost=0.35 rows=1)
    -> Hash
        -> Table scan on t2  (cost=0.35 rows=1)

```
