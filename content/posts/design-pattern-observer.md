---
title: "Observer Design Pattern"
subtitle: ""
date: 2023-10-22T23:11:15+09:00
lastmod: 2023-10-22T23:11:15+09:00
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

## Observer Pattern

한 객체의 상태가 바뀌면, 그 객체를 의존하는 다른 객체에 연락이 가고 자동으로 내용이 갱신되도록 하는 패턴으로 OneToMany 의존성을 정의한다.
여기서 OneToMany 의존성이란, 주제는 상태를 저장하고 제어한다. 따라서 상태가 들어있는 객체는 오로직 하나이다. 반면에 그 상태를 의존하는 Observer 객체는 여러개이다 라는 의미이다.₩₩

주로 프론트엔드 관점에서, observer pattern을 정의한 곳이 많지만 백엔드에서도 observer pattern으로 구현하는 예제 케이스들이 많다.
예를 들어, nodejs의 eventEmitter / logging / socket / distributed system / caching

## observer pattern을 사용하는 이유

- Subject와 Observer 가 서로 독립적으로 느슨하게 결합되어 Subject에 변화가 있어도 Observer에는 영향을 미치지 않는다는 장점이 있다.
- 여러 Observer가 존재하고 있는 상황에서, 새로운 Observer가 추가된다면 단순히 Subject가 `구독`만 해주면 된다. 반대도 마찬가지이다.
- Observer pattern 에서는 Subject 만이 상태를 저장하고 제어함으로써, 리소스를 효율적으로 관리하고 데이터를 제어할 수 있는 곳은 하나이므로 데이터의 일관성을 유지할 수 있다.
- 비동기 작업을 감시하고 있다가, 알림으로 전달하기만 하면 된다.

## 느슨한 결합(loose coupling)

- 주제는 옵저버가 특정 interface를 구현한다는 사실만 안다.
- 옵저버는 언제든지 추가되거나 제거될 수 있다.
- 주제나 옵저버를 다른 곳에서 다른 용도로 사용한다고 해도 문제가 없다. 왜냐하면 느슨하게 결합되어 있기 때문이다.

느슨한 결합은 객체 사이의 상호의존성을 최소화함으로써 변경사항이 생겨도 무난히 처리할 수 있는 유연한 객체지향 시스템을 구축하는 방법이다.

## 예제

예를 들어 NewPost라는 class가 있다고 가정하자. 이 class 는 새로 발행된 게시물을 관리하는데 새로운 게시물이 등록될 때마다 우리는 카테고리별 게시글 수를 업데이트 / 일별 포스팅 수 업데이트 / 해당 게시글을 작성한 유저를 팔로잉하고 있는 사람들에게 나간 알림 수를 업데이트해야 한다고 해보자.

구현보다는 인터페이스라는 디자인 원칙 / 바뀌는 부분은 캡슐화한다는 디자인 원칙을 어기고 대충 짜보면 다음과 같을 것이다.

```typescript
class NewPosts {
  const userIdx = getUserIdx()
  const postIdx = getPostIdx()
  const followingIds = getFollowingIds()

  // 구현에 편의성을 위해 필요없는 인자도 넘겼다.
  postNumByCategory.update(userIdx, postIdx, followingIds)
  postNumByDaily.update(userIdx, postIdx, followingIds)
  notiNumByFollowing.update(userIdx, postIdx, followingIds)

}
```

위와 같이 구성을 하게 되면, 추가적인 통계정보를 추가하거나 삭제할 수가 없다. 왜냐하면 이미 newPosts라는 객체에서 구현되어 있기 때문이다.
추가적으로 update하는 함수는 충분히 바뀔 여지가 있으므로 캡슐화를 해야할 것이다.

```typescript
interface Subjects {
  // Attach an observer to the subject.
  attach(observer: Observer): void;

  // Detach an observer from the subject.
  detach(observer: Observer): void;

  // Notify all observers
  notify(): void;
}

class NewPosts implements Subjects {
  postIdx: number;
  userIdx: number;
  followingIds: number;

  /**
   * The subscription management methods.
   */
  public attach(observer: Observer): void {
    const isExist = this.observers.includes(observer);
    if (isExist) {
      return console.log("Subject: Observer has been attached already.");
    }

    console.log("Subject: Attached an observer.");
    this.observers.push(observer);
  }

  public detach(observer: Observer): void {
    const observerIndex = this.observers.indexOf(observer);
    if (observerIndex === -1) {
      return console.log("Subject: Nonexistent observer.");
    }

    this.observers.splice(observerIndex, 1);
    console.log("Subject: Detached an observer.");
  }

  /**
   * Trigger an update in each subscriber.
   */
  public notify(): void {
    console.log("Subject: Notifying observers...");
    for (const observer of this.observers) {
      observer.update(this);
    }
  }
}

interface Observer {
  update(subject: Subject);
}

class PostNumByCategory implements Observer {
  update(subject: Subject) {
    // update post num by category
  }
}

class PostNumByDaily implements Observer {
  update(subject: Subject) {
    // update post num by Daily
  }
}

class NotiNumByFollowing implements Observer {
  update(subject: Subject) {
    // update noti num by following
  }
}
```

위 처럼 구성하게 되면, 아래와 같이 subject 가 observer를 구독하고 또는 해지하는 것이 용이하다.

```typescript
const newPosts = new newPosts();

const postNumByCategory = new PostNumByCategory();
newPosts.attach(postNumByCategory);

const PostNumByDaily = new PostNumByDaily();
newPosts.attach(PostNumByDaily);

const NotiNumByFollowing = new NotiNumByFollowing();
newPosts.attach(NotiNumByFollowing);

// 만약 새로운 게시글이 생겼다면 단순히 구독중인 observer를 고려하지 않고 notify를 호출하기만 하면 된다.
newPosts.notify();
```
