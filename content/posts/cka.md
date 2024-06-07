---
title: "[Certification] CKA 후기"
subtitle: ""
date: 2024-06-07T20:13:03+09:00
lastmod: 2024-06-07T20:13:03+09:00
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

## 취득 이유

EKS를 써보지도 않았고, 쿠버네티스에 대해 전혀 개념이 존재하지 않아 한번 배워보고자 땄음. SAA를 따고 후회했던 게 덤프만 쥰나게 외우고 보는 거라 사실 왜 땄는지도 모르겠음.
근데 CKA는 덤프가 존재하는 게 아니라, 직접 cli를 통해 트러블 슈팅하고, 어떻게 인프라가 세팅 되었는지 찾고, 복구하고 백업하고 등등 쿠버네티스 서비스를 직접 사용해보면서 익힐 수 있는 거라 더 체감이 되었던 듯.

## 공부 방법

udemy 의 이 [강의](https://www.udemy.com/course/certified-kubernetes-administrator-with-practice-tests/?couponCode=ST21MT60724) 를 들었고 세일기간에 사면 19000원에 살 수 있음.
나는 k8s를 써본적이 없기에 처음부터 강의를 끝까지 천천히 들었고 한달정도 걸렸음. 공부하고 나서 느낀거지만, 강의를 하나하나 정리하면서 모든걸 이해하겠다라는 마음가짐을 가질 필요는 없을 거 같음.
네트워크 섹션에서 나오는 ip 명령어라던지, 직접 namespace 생성하고 리눅스 네트워크 인터페이스 만들고 하는 헛짓거리 안해도 됌(시험에 안나옴, 난 했음)

강의 1.5배속으로 빠르게 보고 제공되는 practice test를 존나게 풀어보는 것이 이득임. 강의를 쥰나게 빨리보고 lightning exam, mock exam, killer sh를 3~4번 반복하면 충분히 합격할 수 있음. 시험이 생각보다 어렵진 않음.
시험에 무조건 출제되는 유형이 있음. ETCD 복구/백업, node 망가졌는데 왜 그런지, network policy, service 등등 무조건 나오는 부분은 꼼꼼히 보고 나머지는 꽤 쉽게 풀 수 있기 때문에 걱정안해도 될 듯.

## 시험 환경, 팁

나는 집에서 봤는데, 제공되는 PSI 프로그램이 졸라 이상함. 갑자기 웹캠이 꺼지고, 이상한 팝업이 쥰나게 뜨고, 감독관은 애매모호하게 말해서 얼탱이 겁나깠음.
그리고 책상에 모든 것을 치웠는데 내 책상 옆에 있는 책을 다 치우라해서 황당함. 결국 다 치우느라 힘빠지고 땀만 흘림. 다음에 시험볼 땐 걍 스터디카페를 가야겠음. (SA 볼때는 안 그랬는데..)
나는 글림쉘모드로 모니터 켜서 맥북을 사용했고, 웹캠이 따로 있어서 모니터 위에 두고 시험봄. 그 외에 모니터는 책상에서 치우라 함. (왜? 꺼져있는데)
캠으로 사방팔방 다 녹화시킴. 책상에 깔아놓은 데스크 매트 밑에 들쳐봐라, 키보드 아래 들쳐봐라, 허브는 뭐냐 쓰는 거냐, 등등 걍 시험볼 때 책상 주변에 뭐가 많다? 걍 스터디 카페 가셈. 거의 이사시킴.
<br>

그리고 killer.sh를 써봐야하는게, 복붙키를 익혀야되기 때문임. 맥북 기준 command + c 가 아니라 control + shift + c,v 임. 그래서 시험 보기 전, 자신의 키보드에 맞게 키보드 세팅을 변경하는 걸 추천
제공해주는 프로그램 내에서, kubernetes document를 firefox로 킬텐데 복사할 때는 마우스 우클릭 추천. 웹 내에서는 ctrl + shift + c 누르면 개발자모드 켜짐.
<br>

쉘에서 clear 존나 연발하다보면, 내가 context를 바꿨는지 안 바꿨는지 까먹을 때가 있음. 웬만하면 한 문제 풀고나서 터미널 clear 시키는 거 추천. 만약 다른 context를 사용해서 문제 풀었으면 틀릴 수도.
프로그램 내에 메모장 있음(난 쓰지 않음), 긴가민가한 문제 flag킬 수 있는 기능있음(해당 문제로 돌아갈 수 있음.)
<br>

아 그리고, 명령 실행하고 hang 걸렸을 때, control + c 아님, control + z 임.
추가로, ssh 접속하는 문제 몇 개 나올텐데, 다음 문제 넘어가기 전에 꼼꼼히 확인하길.
<br>

## 기출문제리스트

기출문제는 나보다 더 잘하는 사람들이 정리한 곳을 추천해드림

[피터의 개발이야기](https://peterica.tistory.com/348)
[dm911](https://velog.io/@dm911/2024-kubernetes-CKA-%EC%9E%90%EA%B2%A9%EC%A6%9D-%EC%B7%A8%EB%93%9D-%ED%9B%84%EA%B8%B0)
[희원이 블로그](https://heewon0704.tistory.com/28)
[dm9111-killer.sh풀이](https://velog.io/@dm911/kubernetes-CKA-study-36-CKA-%EC%8B%9C%ED%97%98%EC%A0%91%EC%88%98-%EB%B0%8F-killer.sh-%ED%92%80%EC%9D%B4)

## 인증

82점 맞았음

<p align='center'>
<img src="/images/cka.png" width="80%"/>
</p>
