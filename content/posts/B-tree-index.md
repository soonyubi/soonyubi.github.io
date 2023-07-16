---
title: "B-Tree index"
subtitle: ""
date: 2023-07-08T21:52:52+09:00
lastmod: 2023-07-08T21:52:52+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: ["mysql", "btree", "index"]
categories: ["database", "mysql"]

featuredImage: "/images/btree.png"
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

## B-Tree Index

`B-Tree` 인덱스는 인덱싱 알고리즘 중 가장 범용적으로 사용되는 알고리즘이다. 여기서 B 는 `Balanced` 를 의미한다. (Binary 아님) B-Tree index는 구조체 내에서 항상 정렬된 상태를 유지한다.

### 구조 및 특징

B-Tree 인덱스는 Root / Branch / Leaf 의 3가지의 노드로 구분된다. 여기서 Leaf 노드는 실제 데이터가 저장된 데이터 파일의 주소를 가르키고 있다.

<p align='center'>
<img src="/images/btree-structure.png" width="80%"/>
</p>

위 그림을 보면, 인덱스는 항상 정렬된 채로 유지되지만, 데이터 파일은 무작위의 순서로 저장된 것을 볼 수 있다. 이 이유는 데이터 파일의 일부 레코드가 삭제되어 빈 공간이 생기게 되면 해당 공간을 재활용하기 때문이다.

### 데이터가 추가,변경,삭제 되면 인덱스는 어떻게 작동하는지?

- 추가

  - 새로운 데이터가 추가되고 해당 데이터를 참조하는 인덱스를 추가할 때 해당 레코드의 key값과 주소정보를 leaf node에 저장한다. 만약, leaf node가 꽉 차서 저장할 수 없을 때는, 리프노드를 분리 시켜야 하는데 이렇게 되는 경우 상위 branch node의 영역까지 확장된다. 이런 이유로, 인덱스를 사용할 경우 write의 작업영역은 cost가 많이 들게 된다.

  - 인덱스 추가로 인해 write 작업이 얼만큼 영향을 받을지를 생각하려면, 테이블의 column 수, column의 크기, index column을 고려해야 한다. 대략적으로 레코드를 추가하는 작업의 cost를 1이라 가정하면 인덱스에 키를 추가하는 cost는 1.5로 생각한다. 따라서 인덱스가 3개가 등록된 테이블의 경우 cost가 대략적으로 (1+1.5\*3) = 5.5 가 된다.

  - mysql 8.0 이상인 경우, InnoDB 엔진을 사용한다. 해당 엔진은 인덱스를 즉시 추가하는 것이 아니라 좀 더 지연시켜 처리할 수 있다. 하지만, PK나 unique index의 경우 중복 체크를 해야하므로 즉시 인덱스를 추가한다.

- 삭제

  - 삭제를 하는 경우 해당 데이터를 가르키는 리프노드를 찾아 삭제마킹을 해주고, 해당 공간은 방치되거나 재활용할 수 있도록 InnoDB storage engine이 알아서 관리해준다.

- 변경

  - 변경의 경우 단순히 index의 key값만 변경하는게 아닌, 해당 데이터를 삭제하고 새로운 데이터를 추가하는 형태로 처리해야 한다.

### Index에 영향을 미치는 요소

- Index column의 크기
  - 위 그림에서 인덱스는 페이지 단위로 관리가 되는 것을 볼 수 있다. B-tree 인덱스에서 각 노드가 자식 노드를 몇 개를 가지고 있는지가 index에 영향을 미칠 수 있는 요소이다. 페이지는 `innodb_page_size` 변수를 조정하여 (4~64KB) 까지 조정할 수 있다.
  - 예를 들어, 인덱스 key의 크기가 16Byte 이고 자식노드의 주소가 12byte라고 한다면, 하나의 인덱스 페이지(16KB)에는 16*1000 / (12+16) = 585개를 저장할 수 있다. 그런데 만약, 인덱스 key의 크기가 32 byte로 늘어날 경우, 하나의 인덱스 페이지에는 16 * 1000 / (12 + 32) = 372개를 저장할 수 있다. <br>
    만약 500개의 record를 읽는 SELECT 쿼리가 존재할 때, 첫번째의 경우엔 하나의 페이지내에서 처리할 수 있지만, 두번째의 경우엔 2개의 인덱스 페이지를 읽어야 처리할 수 있으므로, column의 크기가 인덱스 처리에 영향을 줄 수 있다고 말할 수 있다.
- record 갯수

  - 인덱스를 이용해서 조회하는 것은 cost가 드는 작업이다. 대략 레코드를 읽는 것보다 4~5배 정도 cost가 드는 작업이다. 그러므로, 인덱스를 통해 읽어야할 레코드의 갯수가 실제 레코드의 20~25%를 넘어서면 레코드를 읽는 것이 더 효율적이다. 이러한 작업은 Mysql optimizer가 자동으로 처리한다.

- unique index의 갯수 (=Cardinality, Selectivity)
  - 모든 인덱스의 key가 100개이고, 그 중 unique한 key의 갯수는 10이라 할 때, cardinality = 10 이라고 할 수 있다.
  - cardinality가 높을수록 검색대상이 줄어들기 때문에 속도가 빨라진다. 즉, 전체 index key 갯수 대비 unique key 갯수가 많아야 검색속도가 빨라진다.
  - 예를 들어, 1만건의 데이터가 저장되어 있고, 첫번째 경우엔, unique column 갯수가 10, 두번째 경우엔 unique column 갯수가 1000이라 하자. <br>
    첫번째의 경우엔 carninality = 1000 이고 1개의 데이터를 읽기 위해 999번의 조회를 한 것이다. 두번째의 cardinality = 10이고, 1건의 데이터를 읽기 위해 9건의 조회를 더 한 것이다.

## 인덱스를 어떤 방식으로 이용하는지 ?

### Index Range Scan

```
SELECT * FROM employees WHERE first_name between 'amy' and 'gary'
```

employee 테이블에 first_name 컬럼에 index를 적용하고 위 쿼리를 호출한다면 다음과 같이 검색할 것이다. <br><br>

<p align='center'>
<img src="/images/index_range_scan.png" width="60%"/>
</p>

인덱스를 읽을 위치를 찾는 것을 `index seek` 이라 하고 찾은 순간부터 쭉 읽는 것을 `index scan` 이라 한다.
조건에 만족하는 인덱스들을 찾고 난 후엔 데이터 파일로 부터 random IO 가 일어난다. random IO는 데이터를 읽는 속도에 영향을 미치므로, 인덱스를 이용해서 데이터를 찾는 작업은 비용이 많이 드는 작업이다.

여기서 어떤 경우엔 마지막에 random IO를 통해 데이터 파일에서 데이터를 읽어오지 않아도 되는데 이를 `Covering Index`라 한다.

index seek, index scan 을 확인하는 방법은 다음과 같이 쿼리를 보내면 된다. `SHOW STATUS LIKE '%Handler'`

{{< admonition note "Covering Index" >}}
쿼리 요청 시 인덱스가 걸린 컬럼만으로 쿼리를 완성할 수 있는 걸 의미한다. 예를 들어 (first_name, last_name) 컬럼에 인덱스가 생성되어 있고, 다음과 같이 쿼리를 날릴 경우 데이터 파일에 접근하지 않아도 되므로 성능이 좋다.
`select first_name, last_name from table where first_name between 'amy' and 'gary' and last_name between 'alberts' and 'richard'`
{{< /admonition >}}

### Index Full Scan

인덱스가 생성되긴 했지만, 리프 노드의 처음부터 끝까지 모든 인덱스를 읽을 경우를 의미한다. 그렇다는 건 인덱스를 제대로 이용하지 못했다는 걸 의미하고 다음과 같은 상황에서 발생한다.<br>
(A,B,C) 컬럼에 대해 인덱스를 생성했지만, 쿼리는 B,C 조건을 이용해서 조회할 경우이다. <br>
`index range scan` 보다 성능이 좋지 않지만, 인덱스가 생성된 컬럼만 조회하는 경우 `table full scan` 보다는 성능이 좋다. 왜냐하면, 인덱스의 크기가 테이블의 전체 크기보다는 작기 때문이다.

<p align="center">
<img src="/images/index_full_scan.jpg" width="60%"/>
</p>

### Index Loose Scan

`index skip scan` 은 `index loose scan` 과 비슷하게 작동은 하지만, 중간에 필요하지 않은 값은 Skip 하고 넘어간다. 일반적으로 group by 나 max, min 함수에 대해 최적화를 하는 경우에 발생한다.
// todo : 10장 실행계획 읽고 추가

### Index Skip Scan

인덱스에서 핵심은 값이 정렬되어 있다는 것이고, 그렇기 때문에 인덱스의 순서도 중요하다는 것이다.<br>
employee 테이블엔 (gender, birth_date) 라는 순서로 인덱스가 생성되어 있다고 가정할 때 다음 두 쿼리의 성능은 차이가 발생한다. <br>

```
SELECT * FROM employees WHERE birth_date >= '1965-01-01'
```

```
SELECT * FROM employees WHERE gender = "M" and birth_date >= '1965-01-01'
```

첫번째 쿼리는 `Table Full Scan`으로 처리하고, 두번째 쿼리는 `Index Range Scan`으로 처리한다.
<br>
쿼리를 요청할 때 인덱스에 포함된 컬럼만을 조회하는 경우는 또 상황이 다르다. 예를 들어 첫번째 쿼리와 비슷하지만 인덱스가 생성된 컬럼만을 조회하는 쿼리가 있다고 해보자.

```
SELECT gender, birth_date FROM employees WHERE  birth_date >= '1965-01-01'
```

현재 Mysql 8.0 부터는 optimizer가 birth_date 컬럼만으로 `index skip scan` 을 사용하도록 최적화를 해준다.
따라서 위 쿼리는 `Index Skip Scan`을 사용하여 쿼리를 처리한다.

`Index Skip Scan` 을 사용할 땐 다음과 같은 주의점이 선행된다.

1. 쿼리가 인덱스에 포함된 컬럼만으로 처리가 가능해야함. (covering index)
2. 조건이 포함되지 않은 선행 컬럼의 유니크한 것의 갯수가 적어야 함.

2번의 경우를 예를 들면, (emp_no, dept_no) 로 이루어진 인덱스가 있다고 할 때, dept_no로만 조건을 걸고 조회를 하는 경우, emp_no의 유니크한 갯수가 많기 때문에 skip을 여러번 해야하므로 조회 성능이 좋지 않다.
