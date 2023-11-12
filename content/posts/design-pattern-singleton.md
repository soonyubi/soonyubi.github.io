---
title: "Singleton Design Pattern"
subtitle: ""
date: 2023-11-12T21:28:46+09:00
lastmod: 2023-11-12T21:28:46+09:00
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

## why use singleton

In some cases, creating resources are unnecessarily large, or the consistency of objects can be lost as multiple objects are created. For instance, this can be observed with threads pool, cache, object handling environment variables, and object used for logging purpose.

## singleton vs global variables

- Access to a singleton method achieved through a static method.
- Singleton is initialized the first time when it is used.

- Global variables can be accessible and update it whenever, wherever
- Global variables is initialized when program start

## Singleton pattern

- Ensure that class has just a single instance.
- Other class can generate singleton instance.

### The classic implementation of Singleton

```typescript
class Singleton {
  private static instance: Singleton;

  private constructor() {}
  public static getInstance(): Singleton {
    if (!Singleton.instance) {
      instance = new Singleton();
    }
    return Singleton.instance;
  }
}
```

### The classic implementation of Singleton has drawback

In the classic implementation of singleton may not working correctly in multithreaded environment. This is because if multiple threads attempt to create singleton instance at nearly same time, multiple singleton instance might be created.

## pros and cons of singleton pattern

### pros

- You can be sure that a class has only a single instance.
- You gain a global access point to that instance.
- The singleton object is initialized only when it’s requested for the first time.

### cons

- Violates the Single Responsibility Principle. The pattern solves two problems at the time.
- The Singleton pattern can mask bad design, for instance, when the components of the program know too much about each other.
- The pattern requires special treatment in a multithreaded environment so that multiple threads won’t create a singleton object several times.
- It may be difficult to unit test the client code of the Singleton because many test frameworks rely on inheritance when producing mock objects. Since the constructor of the singleton class is private and overriding static methods is impossible in most languages, you will need to think of a creative way to mock the singleton. Or just don’t write the tests. Or don’t use the Singleton pattern.
