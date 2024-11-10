---
title: "Decorator Design Pattern "
subtitle: ""
date: 2023-10-29T13:28:40+09:00
lastmod: 2023-10-29T13:28:40+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: ["application-development"]

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

## Decorator Pattern

객체가 할 수 있는 행동들이 정의된 상태에서, 객체에 추가적인 행동이 요구되었을 때 객체를 확장하는 것이 아닌 특수한 행동을 하는 객체 래퍼에 넣어서 추가적인 행동을 할 수 있도록 한다.
즉, 기본기능에 추가할 수 있는 기능의 종류가 많은 경우 각 추가 기능을 Decorator에 정의하여 조합함으로써 추가기능을 구현할 수 있다.

예를 들어 푸쉬 알림을 보낼 때 필요한 데이터는 초기에 기획된 데이터보다 늘어날 수가 있다. A,B,C 푸쉬를 보낼 때 기본적인 틀은 비슷하지만, 각각 보내야 할 데이터는 내부적으로 조금씩 다르기 마련이다. 예를 들어, 어떤 푸쉬는 app scheme을 보내야하고, 어떤 푸쉬는 webview를 띄울 url을 보내야하는 그런 경우이다.
이렇게 기본적으로 정해진 틀은 존재하지만, 추가적인 기능이 많아지고 이를 조합해야 할 경우 데코레이터 패턴을 쓰면 동적으로 여러 조합을 구현할 수 있다.

물론, 상속을 이용해서 다양한 기능, 조합을 가진 클래스들을 구현할 수 있다. 다만 상속은 정적이다. 한번 구현되면 런타임에 행동을 변경할 수 없다.

데코레이터 패턴에서는 이러한 사항을 극복하기 위해, 상속 대신 `집합 관계` 또는 `구성`을 활용한다. 이러한 접근방식을 사용하면 연결된 도우미 객체를 다른 객체로 쉽게 대체하여 런타임 때의 컨테이너의 행동을 변경할 수 있다. 객체는 여러 클래스의 행동을 사용할 수 있고, 여러 객체에 대한 참조가 있으며 이 객체들에 모든 종류의 작업을 위임한다.

<p align='center'>
<img src="/images/decorator/decorator-1.png" width="80%"/>
</p>

이렇게 구성하고 나서 email -> facebook -> slack으로 알림을 보내야 한다면 클라이언트는 다음과 같이 구성하면 된다.

```typescript
stack = new Notifier();
if (facebookEnabled) {
  stack = new FacebookDecorator(stack);
}
if (slackEnabled) {
  stack = new SlackDecorator(stack);
}
app.setNotifier(stack);
```

## 구조

```typescript
/**
 * Base Component interface 는 데코레이터에 의해 달라질 수 있는 행동을 정의한다.
 */
interface component {
  operation(): string;
}

/**
 * Concrete Component 는 Component interface의 기본 행동을 정의한다.
 * 해당 클래스는 다양한 변형이 일어날 수 있다.
 */
class ConcreteComponent implements Component {
  public operation(): string {
    return "ConcreteComponent";
  }
}

/**
 * The base Decorator class follows the same interface as the other components.
 * The primary purpose of this class is to define the wrapping interface for all
 * concrete decorators. The default implementation of the wrapping code might
 * include a field for storing a wrapped component and the means to initialize
 * it.
 */
class Decorator implements Component {
  protected component: Component;

  constructor(component: Component) {
    this.component = component;
  }

  /**
   * The Decorator delegates all work to the wrapped component.
   */
  public operation(): string {
    return this.component.operation();
  }
}

/**
 * Concrete Decorators call the wrapped object and alter its result in some way.
 */
class ConcreteDecoratorA extends Decorator {
  /**
   * Decorators may call parent implementation of the operation, instead of
   * calling the wrapped object directly. This approach simplifies extension
   * of decorator classes.
   */
  public operation(): string {
    return `ConcreteDecoratorA(${super.operation()})`;
  }
}

/**
 * Decorators can execute their behavior either before or after the call to a
 * wrapped object.
 */
class ConcreteDecoratorB extends Decorator {
  public operation(): string {
    return `ConcreteDecoratorB(${super.operation()})`;
  }
}

/**
 * The client code works with all objects using the Component interface. This
 * way it can stay independent of the concrete classes of components it works
 * with.
 */
function clientCode(component: Component) {
  // ...

  console.log(`RESULT: ${component.operation()}`);

  // ...
}

/**
 * This way the client code can support both simple components...
 */
const simple = new ConcreteComponent();
console.log("Client: I've got a simple component:");
clientCode(simple);
console.log("");

/**
 * ...as well as decorated ones.
 *
 * Note how decorators can wrap not only simple components but the other
 * decorators as well.
 */
const decorator1 = new ConcreteDecoratorA(simple);
const decorator2 = new ConcreteDecoratorB(decorator1);
console.log("Client: Now I've got a decorated component:");
clientCode(decorator2);
```

## 적용

상속을 사용하여 객체의 행동을 확장하는 것이 어색하거나 불가능할 때 사용하면 좋다. 예를 들어 필요로 하는 추가기능이 한개씩 늘어갈 때마다 2배로 상속클래스를 구현해야 한다거나 불필요한 상속을 할 때 데코레이터 패턴을 사용하면 코드가 깔끔해진다.

어떤 객체를 사용하는 코드를 변경하지 않으면서 추가적인 행동을 객체에 할당할 수 있어야 할 때 사용한다.
