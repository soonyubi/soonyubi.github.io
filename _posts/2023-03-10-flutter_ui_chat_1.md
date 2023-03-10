---
layout: post
title: Flutter UI - 채팅앱을 보고 따라해보기 - (1)
author: soonyubing
description:  
featuredImage: null
img: null
tags: [flutter chatapp]
categories: frontend
date: '2023-03-10 11:17:00 +0900'
# image:
#   src: /assets/img/Chirpy-title-image.jpg
#   width: 1000   # in pixels
#   height: 400   # in pixels
#   alt: image alternative text
---
<style>
r { color: Red }
o { color: Orange }
g { color: Green }
bgr {background-color:Red}
bgy {background-color : Yellow}
bg {background-color: #83f5ef; color : Green}
</style>

![Desktop View](/assets/img/ui1.png){: width="1200" height="700" }

## 1.
`Scaffold`를 사용했고, `Scaffold.AppBar`에는 타이틀, action icon list, 그리고 하단에는 `TabBar`가 들어갔음.

action list 마지막에 더보기 버튼은 `PopupMenuButton`임

각각의 Tabbar에 맞는 페이지를 보여주기 위해, `Scaffold.body`에는 `TabBarView` 를 사용했음.

여기서, `TabBar` 를 사용하기 위해선 `TabController`가 필요한데, `TabController`를 초기화하기 위해서는 `TickerProvider` 가 필요함. `TickerProvider`를 mixin class에서 사용하기 위해 mixin class인 `SingleTickerProviderStateMixin`를 with keyword를 이용해 상속했음.

## 2.

`TabBarView` 에는 floating action button, 3개의 chat message를 볼 수 있음. 이를 위해 `Scaffold를` 사용함.

3개 또는 그 이상의 chat message를 찍어주기 위해 `Scaffold.body` 로는 `ListView.builder` 를 사용했음. 

## 3.

`ListView.builder` 는 다음과 같은 형태의 메세지를 출력하도록 구성했음. `Inkwell` 로 tap이 가능해하고, tap 이벤트 발생 시 채팅 페이지로 이동하는 기능이 필요함. <br>
`Inkwell` 내부에는 `Column` 을 넣고 `Column` 내에는 `ListTile` 과 `Divider` 가 포함됨 
`ListTile.leading` 으로 `CircleAvatar를` 넣어주고, `ListTile.title` 로는 채팅 상대 이름, `ListTile.subtitle` 로는 채팅 마지막 내용을 표시, `ListTile.trailing` 은 마지막 채팅 시간을 보여주도록 구성했음.

