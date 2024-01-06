---
title: "Kotlin 문법"
subtitle: ""
date: 2023-12-03T15:17:56+09:00
lastmod: 2023-12-03T15:17:56+09:00
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

### 싱글턴, companion object

싱글턴 패턴을 구현할 때는 몇가지 제약사항을 통해 구현한다.

- 직접 인스턴스화 하지 못하도록 생성자를 private으로 숨긴다.
- getInstance() 라는 단일 인스턴스를 반환하는 static 메서드를 생성한다.
- 멀티 스레드 환경에서도 유일하게 인스턴스를 반환해야 한다.

구현방법들

- Double Check Locking
- Enum 싱글턴
- Lazy 초기화
- Eager 초기화

코틀린에서는 object 키워드를 통해 싱글턴을 구현할 수 있음. 클래스를 생성함과 동시에 객체를 메모리에 생성함.
생성자를 생성할 수 없음.
상속이 가능함.
다른 클래스안에 생성할 수 있음. 똑같이 객체는 하나만 생성됨.

코틀린에는 static이 없음. 대신 companion object를 제공함.
따라서, companion object를 이용해서 여러 생성자를 가진 싱글턴 클래스를 만들 수 있음.

### Sealed Class

kotlin compiler 는 어떤 자식클래스들이 상속받았는 지 모름. 같은 패키지나 같은 모듈안에 생성된 하위 클래스에 한해서 컴파일러가 해당 서브 클래스들이 상속받았는 지 알게 하기 위해 Sealed Class 를 사용함.
Sealed class 를 사용하면, 상위 클래스에서 상속받은 하위클래스를 when 절에서 빠트렸을 때 컴파일러가 알아채게 할 수 있음.

### 클래스 확장 기능

기존에 코틀린에 정의된 클래스의 메서드들을 확장해서 사용할 수 있음
확장함수에 쓰이는 this는 수신자 객체라고 함

```kotlin
fun String.first() : Char {
    return this[0];
}

fun String.addFirst(char : Char) : String {
    return char + this.substring(0);
}

이렇게 되면 java에서는 Top level로 메서드를 선언할 수 없으니,
public class blah {
  static final function first(String $this$first) : String {
    ...
  }

  static final function addFirst(String $this$addFirst, char : Char ) : String {
    ...
  }
}
```

```
fun MyExample.printMessage(message: String) = println(message);
fun MyExample?.printNullOrNotNull(){
    if(this == null){
        println("null!!!")
    }else{
        println("not null!!!")
    }
}
위 처럼 확장함수는 null 안정성 체크도 할 수 있음
```

### Generic

- star projection
  어떤 타입이 올 지 모르는 경우에 사용
- 변성
