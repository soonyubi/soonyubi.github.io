---
title: "Package Manager"
subtitle: ""
date: 2023-10-09T23:49:40+09:00
lastmod: 2023-10-09T23:49:40+09:00
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

## 패키지 매니저

패키지 매니저는 다음과 같은 기능을 한다.

- 의존성 설치 / 삭제 / 업데이트
- 의존성 관리
- 패키지 배포
- 캐싱
- 보완

대부분의 패키지 매니저가 기능적으로 비슷하게 보일 수 있으나, 내부적으로는 매우 다르다. npm/ yarn classic의 경우 `node_modules`라는 폴더안에 flat하게 저장되는 구조로 dependency를 설치했다. 이러한 방식은 비효율적이고 깨지는 문제가 생길 수가 있다.

pnpm, yarn berry는 이러한 문제를 해결했고, 많은 패키지 매니저들 속에서 우리의 워크플로우에 알맞게 선택해야 할 순간이 왔다.

## NPM

### npm의 비효율적인 의존성 검색

npm 은 파일 시스템을 이용해서 의존성을 관리한다. node_modules라는 폴더 내에 모든 의존성을 flat하게 관리하는 게 특징인데 이는 비효율적으로 동작한다.

예를 들어 `require('react')` 라는 패키지를 불러오는 과정에서, npm은 해당 패키지를 찾기 위해 계속 상위 디렉토리의 node_modules폴더를 탐색한다. 패키지를 바로 찾지 못할수도 있고 중간에 실패하기도 한다.

### 비효율적인 설치

npm에서 node_modules는 매우 큰 공간을 차지한다. 추가로, 매우 큰 디렉터리이기 때문에 설치가 유효한 지 검증하기가 어렵다.

### 유령 의존성

npm과 yarn 1.x은 중복되어 설치되는 패키지를 막기 위해 Hoisting 방식을 사용한다.

<p align='center'>
<img src="/images/package_manager/hoisting.png" width="80%"/>
</p>

위 그림에서 보면 A(1.0) / B(1.0)은 두번씩 설치되므로 공간을 낭비하므로 오른쪽 그림처럼 flat하게 한번만 설치하게 된다. 근데 이렇게 되면 직접 설치하지 않은 패키지를 require할 수 있고 이를 `Phantom Dependency`라고 한다. B라는 패키지를 설치하지 않았지만, 사용할 수 있게 되다가, A라는 패키지가 필요없어져 삭제하게 될 경우, B라는 패키지를 사용할 수 없게 되는 문제가 발생한다.

## Yarn 1.x

yarn은 npm의 비효율적인 설치를 해결하기 위해 등장했다. yarn은 npm 기반으로 설계했지만, 다음과 같은 기능을 추가하였다.

- native monorepo 지원 (npm7.x 이후 버전에서도 지원한다.)
- cache-aware 설치
- 오프라인 캐싱
- lock files

## Yarn Berry

## Pnpm
