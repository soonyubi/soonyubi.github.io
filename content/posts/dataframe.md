---
title: "Dataframe"
subtitle: ""
date: 2023-08-28T23:55:51+09:00
lastmod: 2023-08-28T23:55:51+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: []

featuredImage: /images/dataframe/dataframe.png
featuredImagePreview: /images/dataframe/dataframe.png

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

## Dataframe Basic

- dataframe의 각각의 column은 `series`임
- axis=1은 x축, axis=0는 y축임
- dataframe의 column index는 series의 name과 동일

### properties

- `shape` : (n_row,n_columns)
- `index` : axis=0, RangeIndex
  - df.index -> RangeIndex(start=0,stop=30000,step=1)
- `columns` : axis=1, equals to series names
- `axes` : [index, columns] 꼴
- `dtypes` : 각 column의 type

### creation

```python
df = pd.Dataframe({
  # series_name : array of object
  "id":[1,2],
  "store_nbr":[1,2],
  "family":['POURTRY','PRODUCT']
})

df = pd.read_csv("some_csv_file.csv")

```

[pandas_read_csv_document](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

## Exploring Dataframe

- `head` : get first n rows of df
- `tail` : get last n rows of df
- `sample` : get random sample
  - df.sample(n,random_state) : random_state를 지정해서 같은 랜덤값을 가져올 수 있음
- `info` : column 리스트, 각 column의 type, memory 사용량, index 정보를 가져올 수 있음
  - `show_counts=True` parameter로 non-nullable data의 갯수를 알 수 있음
- `describe` : count / mena / max / 25% / 50% / 75% / min / std 같은 간단한 통계정보를 각각의 column에 대해 구할 수 있음

## Accessing & Dropping Data

### Accessing Data

- dataframe의 각 column은 [] / .(dot) notation 을 통해 접근할 수 있음
- df['some_column']은 series이므로 nunique() / mean() / value_counts() / sum() / round() 같은 메서드를 사용할 수 있음
- multiple column access : [['col1','col2']]
- iloc accessor
- loc accessor
  - df.loc[:,'col1'] 과 df.loc[:,['col1']] 의 결과는 다름, 전자는 series / 후자는 dataframe

### Dropping Data

- df.drop([0],axis=0) &rarr; 0 row를 지운다는 의미
- df.drop(range(5),axis=0) &rarr; 0~4rows를 지운다는 의미

### Identifying Duplicate Data

- df.nunique() : 각 column에 대응하는 데이터 중 unique한 것들의 갯수
  - df_nunique(subset=['col']) 을 통해 원하는 column만 구할 수 있음
- df.duplicated() : row index가 큰 것이 작은 것과 데이터가 동일할 경우 True
  - subset\_['col'] 을 통해 원하는 column만 확인 가능
- df.drop_duplicates() : df.duplicated()가 True인 것들을 제거
  - df.drop_duplicates(subset = 'product', keep='last', ignore_index = True) &rarr; product 컬럼에 대해, 중복은 앞에서 부터 지우며, 중복을 지우고 난 후 row index는 0부터 1씩커지도록 구성하란 의미

## Blank & Duplicate values

- df.isna() : 각 data에 대해 NA면 True
  - df.isna().sum() : 각 column에서 black data 갯수를 구함
- df.fillna() : NA 값을 특정 값으로 채움
- df.dropna() : NA값이 포함된 row를 모두 지움

## sorting & filtering

### filtering

- df.loc[df['col']==='a'] 와 같이 loc 내부 조건에 만족하는 데이터들만 뽑아서 dataframe으로 리턴
- `df.query('family in ['ada','asda'] and sales > 0')`
  - 다음과 같이 변수명을 query 문 안에 쓸 수 있음

```python
avg_sales = df.loc[:,'sales'].mean()

df.query('@avg_sales > 0')
```

### sorting

`sort_index()`를 이용해 dataframe을 index기준으로 sort할 수 있음. axis= 0이 기본값임.

```python
- df.sort_index()
  - axes= 0, row index를 기준으로 sort
- df.sort_index(axis=1)
  - column명을 사전순으로 정렬
- df.sort_index('column1')
  - column1에 들어있는 값을 기준으로 정렬
- df.sort_index(['column1','column2'], inplace=True)
  - column1, column2 기준으로 정렬하고, 해당 데이터프레임 정렬된 것으로 교체
```

## Modifying Columns

### rename columns

df.columns / df.rename() 메서드를 이용해 column 명을 바꿀 수 있음

```python
df.columns = ['a','b','c']
<=>
df.rename(columns={'A':'a','B':'b','C':'c'})

df.columns = [col.upper() for col in df.columns]
<=>
df.rename(columns = lambda x: x.upper())
```

### reordering columns

```python
df.reindex(labels=['column1','column2','column3'])

```

### column creation

```python
df['new_column'] = df['column1'] + df['column2']
df['new_column'] = df['column1'] != df['column2']
```

numpy의 `select`를 이용해 여러개의 조건에 해당하는 새로운 column을 생성할 수 있다.

```python
conditions = [
  df['col1'] != df['col2'],
  df['col2'] != df['col3'],
  df['col3'] != df['col1'],
]
choices = [
  'col1 and col2 are not match',
  'col2 and col3 are not match'
  'col3 and col1 are not match'
]
df['new_column'] = np.select(conditions, choices, default='not in condition')
```

map method를 이용해, 새로운 column의 데이터를 생성할 수 도 있다.
`dataframe['new_column'] = dataframe['col1'].map({'value':'new_value'})`

`df.assign(col1 = df['column1'] + df['column2'])`

## Pandas DataTypes

- bool
  - numpy : boolean True/False, 8bit
  - pandas : nullable boolean True/False, 8bit
- int64
  - numpy : whole numbers, 8/16/32/64 bit
  - pandas : nullable whole number, 8/16/32/64 bit
- float64

  - numpy : whole decimal number, 8/16/32/64 bit
  - pandas : nullable whole decimal number, 32/64 bit

- object

  - numpy : any object
  - pandas
    - string : only string or text
    - category : 반복되는 문자열 같은 것을 숫자에 매핑하여, 메모리 효율을 올림, enum 같은 거

- time series
  - datetime
    - 특정 시각 (e.g. January 4, 2015, 2:00:00 PM)
  - timedelta
    - 2개의 시간 사이의 duration
  - period
    - span of time (a day, a week etc)

## Memory Optimization

dataframe은 메모리에 저장되므로, 메모리 최적화하는 것이 large dataset에 대해 효과적으로 작동함.

`df.memory_usage(deep=True)` 로 메모리양을 확인할 수 있음

1. 필요없는 column은 drop (필요없는 column을 읽는 것도 피하는 것이 좋음)
2. datatype을 가능하다면, int64 / datetime으로 변환
3. 숫자가 어디까지 커질 지 미리 예상해서, 숫자 타입을 downcasting
4. category로 분류할 수 있는 데이터는 categorical datatype으로 지정

- `df.astype({'column':'category'})`

## Grouping Columns

### Aggregate dataframe

series처럼 dataframe에도 aggregate 관련함수를 사용할 수 있다.

```python
df.loc[,['column1','column2']].sample(100).sum().round(2)
df.loc[,['column1','column2']].sample(100).mean().round(2)
```

여러개의 column을 각각 aggregate 하거나, column을 grouping 하고 aggregate하고 싶다면? -> `df.groupby()`

```python
df.groupby(['col1','col2'])[['col3','col4']].sum()
-->col1,col2순서로 데이터를 그룹지어서 [column3,column4] 에 해당하는 sum을 구할 수 있음.
--> 이때 col1,col2는 multi-index임
--> multi-index를 풀려면 다음과 같이 as_index라는 argument를 groupby에 전달해주면 됨
df.groupby(['col1','col2'], as_index = False)[['col3','col4']].sum()


```

## multi-index

multi-index는 aggregate function을 통해 생기는 index임. tuple list로 저장됨

<p align='center'>
<img src="../../static/images/dataframe/multi-index.png" width="70%" height="50%" title="multi-index">
</p>

multi-index를 접근하는 방법은 기존 dataframe과 차이가 있음.
loc[] accessor를 통해 접근하는데 다음과 같이 작동함.

<p align='center'>
<img src="../../static/images/dataframe/multi-index-access
.png" width="70%" height="50%" title="multi-index">
</p>
<p align='center'>
<img src="../../static/images/dataframe/multi-index-access2
.png" width="70%" height="50%" title="multi-index">
</p>

### modifying multi-index

<p align='center'>
<img src="../../static/images/dataframe/multi-index-modify
.png" width="70%" height="50%" title="multi-index">
</p>
