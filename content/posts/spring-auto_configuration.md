---
title: "[Spring] Auto Configuration"
subtitle: ""
date: 2024-06-24T13:46:43+09:00
lastmod: 2024-06-24T13:46:43+09:00
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

## Auto Configuration ?

auto configuration은 어플리케이션의 classspath 나 bean을 기반으로 필요한 설정을 자동으로 구성하는 것을 의미한다.
이를 이용하면, application.yaml 에 db connection 과 관련된 property를 정의하고 db dependency를 설치하는 것으로 간편하게 db에 대한 접근을 쉽게할 수 있다.

auto configuration은 `@SpringBootApplication` 어노테이션에 포함된 `@EnableAutoConfiguration` 어노테이션에 의해 활성화가 된다.

## 작동원리

1. @EnableAutoConfiguration 에 의해 활성화

2. `META-INF/spring.factories`에 정의된 것을 보고 어떤 클래스들을 자동으로 구성할 지 결정한다.

3. 조건부로 구성
   `@Conditional~~` annotation에 의해 특정 클래스가 classpath에 존재할 때만 활성화를 시키거나, 존재하지 않을 때만 활성화를 시킨다.

아래는 RedisAutoConfiguration 클래스의 모습인데, 첫번째 이미지는 redis에 대한 종속성을 추가하지 않은 상태이고, 두번째는 redis에 대한 종속성을 추가한 후에 대한 이미지이다.

<p align='center'>
<img src="/images/spring/auto-configuration-1.png" width="80%"/>

<img src="/images/spring/auto-configuration-2.png" width="80%"/>
</p>

`@ConditionalOnClass(RedisOperations.class)` 의 의미는 RedisOperation 클래스가 classpath 에 있을 때만 활성화한다는 의미이다.

4. 설정파일
   AutoConfiguration 클래스는 application.yaml(properties) 파일을 읽어 특정 클래스의 설정을 커스터마이징한다.

5. 빈 생성 및 초기화
   AutoConfiguration 클래스는 정의한 조건을 만족할 경우, 빈을 등록하고 초기화한다.

6. 우선순위
   AutoConfiguration 클래스에 정의된 빈이 있지만, 사용자가 빈을 직접 정의하는 경우엔 사용자가 직접 정의한 빈이 우선 사용된다. 이를 통해 구성 설정 오버라이딩을 쉽게 할 수 있다.

   또한 Auto ConfigurationOrder를 사용해서 구성 클래스의 적용 순서에 대한 제어도 가능하다.
