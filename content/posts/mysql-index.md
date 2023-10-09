---
title: "optimization index"
subtitle: ""
date: 2023-10-09T12:14:01+09:00
lastmod: 2023-10-09T12:14:01+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: ["database"]

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

## Introduction

다들 잘 알고 있다싶이, 인덱스는 데이터를 효율적으로 빠르게 찾을 수 있는 방법을 의미한다.
인덱스는 한번 생성되면 추가적인 조치가 필요없지만, 쿼리 플래너가 좋은 결정을 내릴 수 있돌고 주기적으로 `ANALYZE` 명령을 실행하여 통계를 업데이트하는 것도 좋은 방법이다.

큰 테이블에 인덱스를 생성하게 되면 시스템은 테이블과 인덱스를 동기화하는 과정이 필요하다.
이 과정에서 DB 엔진마다 다르겠지만, 테이블에 대한 작업이 느려질 수 도 있다.

## Index Type

이 포스팅은, postgres 공식문서를 보고 작성한 글이라 아래와 같이 5개의 인덱스 타입을 설명한다. mysql 의 경우 `Gist` / `GIN` / `BRIN` 의 index type은 없고 `FULLTEXT` / `R-Tree` 같은 별도의 인덱스 타입을 지원한다.

### B-Tree

B-tree는 특정 기준으로 정렬된 순서를 가진 채로 저장된다. 그렇기 때문에 동등비교나 범위쿼리에 대해 효과적으로 쿼리를 수행할 수 있다. LIKE / ~ 같은 연산자와도 인덱스를 사용할 수 있지만, 예외사항이 존재한다. 예를 들어 `column like 'foo%'` `column ~ '^foo'` 는 가능하지만, `column like '%bar'`는 가능하지 않다.

### Hash

해쉬 함수를 이용해서 데이터를 인덱싱하는 방법이다. 해쉬 함수는 임의의 길이의 입력을 받고 고정된 길이를 출력한다. 여기서 출력되는 값을 hash code 라고 하는데 이 값을 이용해 데이터를 조회한다.

해쉬 인덱스를 잘 사용하면 평균 조회시간은 O(1)이고, 공간도 비교하고자 하는 쿼리에 따라 효율적으로 사용할 수 있다.

하지만 별도의 해쉬 테이블을 구성해야 하므로, 메모리 낭비로 이어질 수 있는 단점이 있다.(b-tree 인덱스에 비해 해쉬 테이블의 크기는 작은 편이다.) 해쉬충돌을 피하기 위한 추가적인 연산이 들어갈 수 있다. 범위 쿼리에는 효과적이지 않다는 단점이 있다.

mysql의 innodb 엔진에서 `global innodb_adaptive_hash_index` 옵션을 켜서 자주 사용되는 컬럼에 대해 해쉬 인덱스를 사용할 수 있긴 하지만, 기본 해쉬 자료구조에 계속 쌓여가는 해쉬 데이터를 제어할 수 없기 때문에 나중에 테이블 drop 하는 경우 문제가 생길 수 있다.

### Gist(postgres)

// todo desc

### GIN(postgres)

// todo desc

### BRIN(postgres)

// todo desc

### FULLTEXT(mysql)

// todo desc

### R-Tree(mysql)

// todo desc

## MultiColumn Index

multicolumn index는 다음과 같은 상황에서 효과적으로 사용된다. 선행되는 컬럼의 동등 비교
