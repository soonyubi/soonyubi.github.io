---
title: "Strategy_pattern"
subtitle: ""
date: 2023-07-16T22:36:39+09:00
lastmod: 2023-07-16T22:36:39+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: ["strategy-pattern"]
categories: ["design-pattern"]

featuredImage: "/images/design_pattern_strategy_banner.png"
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

## strategy pattern 이란?

Strategy Pattern은 객체내에서 행동을 바꿀 수 있고, original context class 내에서 서로 상호교환 할 수 있는 패턴을 의미합니다. 즉, 알고리즘을 캡슐화하여 변경가능하도록 해주는 디자인 패턴을 의미합니다. <br><br>

<p align='center'>
<img src="/images/strategy_pattern.png" width="80%"/>
</p>

위 구조에서 가장 상위에 있는 class를 context class라 할 때, context class의 역할은 strategy interface를 참조하는 것입니다. strategy interface 는 모든 견고한 strategies에 공통되는 부분이고, 견고한 strategy는 어떤 행동을 해야할 지를 결정하는 부분입니다.
이렇게 디자인을 구성한다면, context class는 각 strategy가 어떤 역할을 하든 참조하는 strategy가 그 행동을 사용하도록 로직만 구성하면 됩니다.

## 왜 strategy pattern을 사용해야 하는지?

최적의 길을 알려주는 네비게이션 앱이 있다고 하겠습니다. 이 앱은 현재 자동차로 목적지까지 갈 수 있는 최적의 루트를 계산하고 그 루트를 앱에 보여줍니다.
<br>

```javascript
class Navigator {
  public routeForCar(A,B) {
    ... calulate optimal route when user use car
  }
}
```

<br>
앱이 커짐에 따라, 많은 사용자가 들어오고, 사용자들은 다양한 기능을 요구하였습니다. 걸었을 때 최적의 루트를 알고 싶은 사용자, 사이클을 탔을 때 최적의 루트를 알고 싶은 사용자, 대중교통을 이용했을 때 최적의 루트를 알고 싶은 사용자... 그럴 때마다 Navigator 앱은 다음과 같이 각 사용자를 위한 기능을 추가해야 합니다.
<br><br>

```javascript
class Navigator {
  public routeForCar(A,B) {
    ... calulate optimal route when user use car
  }
  public routeForWalker(A,B) {
    ... calulate optimal route when user use walking
  }
  public routeForPublicTransport(A,B) {
    ... calulate optimal route when user use transport
  }
  ...
}
```

다음과 같이 상속을 하고, context class에 정의된 메서드를 overriding 하게 되면, 각 다른 동작을 할 수 있긴 하겠지만 코드를 재사용할 수 없을 뿐더러, 유지보수하기도 힘들어집니다.

```javascript
class Navigator {
  public calculateOptimalRoute(A,B){

  }
}

class Walker extends Navigator{}

class Car extends Navigator {}

class PublicTransport extends Navigator {}
```

따라서, Navigator 클래스는 interface를 참조하고, 행동이 바뀔 수 있는 부분들은 interface를 각각 구현하여 코드를 변경하는 과정에서 의도치 않게 발생하는 일을 줄이면서 시스템의 유연성을 향상시킬 수 있습니다.

이런 방식을 적용하기 위해선 바뀌는 부분과 바뀌지 않는 부분을 생각해야 합니다. 모든 사용자는 `출발지와 목적지를 설정한다`는 행동은 바뀌지 않기 때문에 해당 기능은 모든 strategy class가 재사용할 수 있습니다. 각 사용자가 이용하는 방법에 따라 `최적의 루트를 계산하는 방식엔 차이`가 있기 때문에 그러한 행동은 각 strategy class에서 구현하면 되겠습니다.

```javascript
class Navigation {
  private strategy: Strategy;

  constructor(strategy: Strategy) {
    this.strategy = strategy;
  }

  public setStrategy(strategy: Strategy) {
    this.strategy = strategy;
  }

  public doSomethingBussinessLogic(): void {
    // input start point and end point
    this.strategy.setStartAndEndPoint("a", "b");
    const result = this.strategy.doAlgorithm(["a", "b", "c", "d", "e"]);
    console.log(result.join(","));
  }
}

interface Strategy {
  setStartAndEndPoint(start: string, end: string);
  doAlgorithm(data: string[]): string[];
}

class User implements Strategy {
  private startPoint: string;
  private endPoint: string;
  doAlgorithm(data: string[]): string[] {
    return data;
  }
  setStartAndEndPoint(start: string, end: string) {
    this.startPoint = start;
    this.endPoint = end;
  }
}

class Walker extends User {
  doAlgorithm(data: string[]): string[] {
    return data.sort();
  }
}

class Car extends User {
  doAlgorithm(data: string[]): string[] {
    return data.sort();
  }
}

class PublicTransport extends User {
  doAlgorithm(data: string[]): string[] {
    return data.reverse();
  }
}

const navigation = new Navigation(new Walker());
navigation.doSomethingBussinessLogic();

```

## strategy pattern을 적용시킬만한 곳

- 런타임 시, 다양한 알고리즘을 적용시킬 클래스를 구현해야 하는 경우
- 무수히 많은 클래스가 존재하는데, 그 클래스들이 특정행동을 제외하고는 비슷한 기능을 하는 경우

## strategy pattern의 장단점

- 장점
  - `open/closed principle`을 따릅니다. 즉, 새로운 전략을 추가할 때, context class를 변경하지 않고도 새로운 전략을 추가할 수 있습니다.
  - `Inheritance`를 `Composition`으로 대체할 수 있습니다.
  - 비지니스로직에서 해당 알고리즘을 사용할 때, 구현체를 분리시킬 수 있습니다.
- 단점
  - 적절한 전략을 사용하기 위해 전략들간의 차이를 알아야 할 필요가 있습니다.
  - `lambda`같은 익명함수로 이러한 전략을 구현하여, 굳이 interface, class를 생성하지 않고 구현할 수 있습니다.
  - 적용할 strategy pattern 이 적고, 자주 바뀌지 않는다면 굳이 코드를 복잡하게 만들 필요는 없습니다.
