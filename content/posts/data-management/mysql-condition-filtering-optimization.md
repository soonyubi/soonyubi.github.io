---
title: "Mysql Condition Filtering"
subtitle: ""
date: 2023-10-08T19:03:15+09:00
lastmod: 2023-10-08T19:03:15+09:00
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

## Condition Filtering

join 과정에서, prefix rows는 한 테이블에서 처리한 행들이 join에 쓰이기 위해 전달되는 행을 의미한다.
일반적으로 optimizer는 row의 조합의 수를 증가시키지 않는 방향으로 prefix count가 낮은 테이블을 join의 앞에 두려고 노력한다.

조건 필터링 없이는, prefix rows는 where절에 의해 선택된 행의 예상 수를 기반으로 한다.
반면에, 조건 필터링을 사용하게 되면, 옵티마이저가 이전에 고려되지 않은 where 절의 다른 조건을 사용할 수 있게 해주므로 prefix rows count를 향상시킬 수 있다.

`EXPLAIN`을 보면, `rows` column은 access method에 의해 계산된 row estimate를 의미하고, `filtered` column은 백분율을 의미하는데, 100은 필터링이 발생하지 않았음을 의미한다. 예를 들어 rows : 200 이고, filtered는 100일 경우 200개의 rows를 prefix rows로 읽었다는 것을 의미하고, rows :200 / filtered : 20 이면, 200 \* 20% = 40 개의 rows를 prefix rows로 읽었다는 것을 의미한다.

조건필터링에 기여하는 요소는 다음과 같다.

1. 현재 테이블을 참조해야 함.

```mysql
select * from users join orders on users.id = orders.id where users.age > 25;

users.age는 users라는 현재 테이블을 참조하는 조건임
```

2. 조인 순서에서 이전 테이블의 상수 값 또는 값에 의존해야 함.

```
SELECT * FROM users JOIN orders ON users.id = orders.user_id WHERE orders.order_date > '2022-01-01'

위와 같이 2022-01-01 같은 상수 값에 의존해야 함
```

3. access method 가 이미 고려되지 않았어야 함.

- 예를 들면, users 테이블에 age 컬럼에 인덱스가 있고, 옵티마이저가 이 인덱스를 사용하여 users.age > 25 조건을 처리한다고 가정해. 이 경우, 옵티마이저는 이미 age에 대한 조건을 고려하고 있으므로 다른 방식으로 추가 필터링을 적용하지 않을 거야.

- 그런데, 만약 users 테이블에 name 컬럼이 있고, 쿼리에 WHERE users.name LIKE 'J%' 조건이 추가되었는데, name에 대한 인덱스가 없다면, 이 조건은 접근 방법에 의해 이미 고려되지 않았다고 볼 수 있어. 따라서 옵티마이저는 이 조건을 추가로 필터링 추정에 사용할 수 있을 거야.

옵티마이저가 조건 필터링의 성능을 과대평가하면, 사용하지 않은 경우보다 성능이 나빠질 수 있다. 그럴 때는 다음과 같은 기법을 고려할 수 있다.

- 인덱스가 없다면, 옵티마이저가 컬럼 값의 분포를 알 수 있도록 인덱스를 생성한다.

- 히스토그램 정보가 없다면, 히스토그램을 생성할 수 도 있다.

- 조인 순서를 변경한다. `STRAIGHT_JOIN` 같은 것을 hint로 줄 수 있다.
