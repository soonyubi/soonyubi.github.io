---
title: "Sharding"
subtitle: ""
date: 2023-11-20T00:16:54+09:00
lastmod: 2023-11-20T00:16:54+09:00
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

## Why needs sharding?

When a large amount of data accumulates in a single database, not only does the storage capacity increases, but performance algo degrades. Sharding can be used to distribute the traffic burden on the database, thereby enhancing performance. An important aspect of sharding is efficiently distributing the data.

## What is sharding?

- distributing and storing data with same table schema across multiple databases
-
