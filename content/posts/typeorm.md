---
title: "Typeorm을 써보면서"
subtitle: ""
date: 2023-05-29T23:25:32+09:00
lastmod: 2023-05-29T23:25:32+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: ["backend"]

featuredImage: /images/typeorm.png
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

## 개요

최근 한달 동안 한 스타트업에서 인턴 근무를 하며, typeorm을 써본 후기를 말하고자 한다. 우리 회사는 nest.js + typeorm 쓰고 있고, 기존에 레거시 코드를 typeorm 으로 변환하면서 많은 고충도 느끼고 있지만, 꽤 괜찮은 orm이라는 생각도 들고 있다. 그래서 내가 typeorm을 쓰면서 느꼈던 괜찮은 점과 아쉬운 점에 대해 포스트하고자 한다.

## 장점

1. `QueryBuilder` 를 사용해서 쿼리의 반복을 줄일 수 있다.

다음과 같은 쿼리가 있다고 가정하자.

```
SELECT *
FROM A
LEFT JOIN B on A.Id = B.Id
LEFT JOIN C on A.Id = C.Id
WHERE A.ID = 3

SELECT *
FROM A
LEFT JOIN B on A.Id = B.Id
LEFT JOIN C on A.Id = C.Id
WHERE B.ID = 3
```

위 쿼리 2개를 날릴 때, 2개의 쿼리를 각각 작성하거나, 아니면 View를 생성해야 할텐데, typeorm에서는 다음과 같이 쿼리를 생성하고 재사용할 수 있다.

```javascript
const query = await aRepository
  .createQueryBuilder("a")
  .leftjoin(B, "b", "a.Id=b.Id")
  .leftjoin(C, "c", "c.Id = a.Id");

const select1 = await qeury.where("a.Id = 3").getMany();
const select2 = await query.where("b.Id = 3").getMany();
```

이 기능이 다른 orm에서도 지원하는 건 알고 있지만, 난 typeorm을 쓰면서 orm을 처음 써봤기 때문에 가장 인상적이었다.

2. transaction을 지원한다.

typeorm에서 transaction을 사용하기 위해선, `queryRunner` 나 `dataSource`를 이용해야 한다.

내가 회사에서 사용한 방법은 다음과 같다.

```javascript

private dataSource : DataSource;
...
async function(){
  await dataSource.manager.transaction(async(entityManager : EntityManager)=>{
    await transactionalEntityManager.save(users)
    await transactionalEntityManager.save(photos)
  })
}
```

transaction을 구성할 때, isolation level을 지정할 수 도 있다. `queryRunner`로 구성해보고 싶기도 했지만, 아직까지 구현이 잘 되지 않아 어떻게 해야할 지 고민중이다.

3. caching 이 default로 된다.
   cache를 무조건 쓰는게 좋은 건 아니지만, 어쨋든 cache 기능이 있기 때문에 같은 쿼리를 여러번 날리게 되면 더 빠른 속도로 쿼리를 날릴 수 있다.
   cache 를 사용하기 위해선 dataSource option 값에 추가해주기만 하면 된다.
   추가로 cache를 얼마나 유지할 것인지 시간을 명시할 수 도 있다.

## 단점

1. 공식문서에 설명이 너무 부족하다.

다음과 같이 page, limit을 걸어서 쿼리를 날리려고 했는데 에러를 마주하고 해결하는 데 오래걸렸다. 소스를 뒤져보니, 주석에 left join을 할 때는 `limit `대신 `take`를 `offset`대신 `skip`을 사용하라고 나와 있었다.

```
# it is not working properly
.createQueryBuilder()
.leftjoin()
.leftjoin()
.leftjoin()
.offset(page)
.limit(rows)

.createQueryBuilder()
.leftjoin()
.leftjoin()
.leftjoin()
.offset(page)
.take(rows)
```

2. typeorm maintainer가 소스 관리를 포기한 상태이다.

현재까지도 많은 이슈가 계속해서 올라오고 있다. 나 또한 작업을 하면서 다음과 같은 쿼리를 날릴 때 이슈가 있었다.

```javascript
.where("a.isUse = :isUse",{isUse = 'Y'})
.where("b.isUse = :isUse",{isUse = 'Y'})
.where("c.isUse = :isUse",{isUse = 'N'})
```

위 쿼리가 정상적으로 동작하면, `a.isUse='Y' and b.isUse = 'Y' and c.isUse='N'` 이어야 하는데, `a.isUse='N' and b.isUse = 'N' and c.isUse='N'`
와 같이 가장 마지막에 부여된 값을 기준으로 쿼리를 날리고 있었다.

위 문제뿐만 아니라 아직도 해결되지 않은 이슈가 많이 있다.

## 마무리

아직 부족하지만, 내가 한달동안 typeorm을 사용하면서 느꼈던 점을 간략하게 설명했다. 비록 내가 아직 사용하지 못한 많은 기능이 존재할 거라 생각하고, typeorm을 사용한 경험을 바탕으로 다른 orm도 사용해보면서 비교를 해봐야겠다는 생각을 했다.
