---
layout: post
title: mysql 기본 함수 정리본
author: soonyubing
description: 필요할 때마다 추가 해놓을게요. 
featuredImage: null
img: null
tags: mysql
categories: mysql
date: '2023-02-13 11:17:00 +0900'
# image:
#   src: /assets/img/Chirpy-title-image.jpg
#   width: 1000   # in pixels
#   height: 400   # in pixels
#   alt: image alternative text
---
<style>
r { color: Red }
o { color: Orange }
g { color: Green }
bgr {background-color:Red}
bgy {background-color : Yellow}
bg {background-color: #83f5ef; color : Green}
</style>

## 기본

1. IF (expression, 참 일 때 값, 거짓일 때 값)
2. IFNULL(exp1, exp2) exp1이 not null이면 exp1, null이면 exp2 리턴
3. LENGTH(문자열) 

4. 집계 함수 (sum, count, min, max ,avg )에 대해서 조건을 걸 때는 <o>having</o> 을 사용한다.

## 수학
1. CEIL : 소수점 올림
2. FLOOR : 소수점 내림
3. Truncate(23.933123, 3) : 23.933 
4. ROUND(23.0845,4) : 23.085
5. ABS()
6. MOD()
7. MEDIAN() : ORACLE에만 존재 <BR>
- MYSQL에서 해당 기능을 제공하고 싶다면 다음과 같이 쿼리
```mysql
select percent_rank() over (order by [column]) as prank
from table
where prank=0.5
```

## 문자열
1. CONCAT(str1, str2, ....) 모든 문자열을 합쳐서 리턴
2. CONCAT_WS(구분자, str1, str2, str3....) 문자열을 구분자를 사이에 넣어 리턴
3. FIELD('b','a','b','c') -> 첫번째 문자열을 기준으로 해당 문자열이 위치하는 순서를 찾는다. 없으면 0
4. LOCATE('abc', 'ababcDEFabc') : 처음 abc가 나오는 위치 : 3 - mysql은 인덱스가 1 기준이다. 
5. lower() / upper()
6. replace('backend','back','front') :frontend
7. trim() : 양쪽 공백제거 
8. reverse('abcde') : edcba
9. REPEAT('abc',3) : abcabcabc
10. LPAD(123,5,'0') : 00123 : 
LPAD(col, 원하는 길이, 채울 문자열)
11. substring(col, 시작위치, 길이)
: substring('소프트웨어 마에스트로',5,5) -> ' 마에스트' 
12. substring_index(col, 구분자, count) 
- substring_index("www.abc.com",'.',2) -> www.abc
- substring_index("www.abc.com",'.',1) -> www
13. 어떤 컬럼에 들어있는 값 중에서 숫자만 찾기
```mysql
SELECT *
FROM TABLE
WHERE DATA REGEXP ('^[0-9]+$')
```

## 날짜
1. DATE_ADD(NOW(), <r>INTERVAL</r> 1 MONTH ) : 현재시각에 1달 더하기, MONTH를 포함해 SECOND/ MINUTE / HOUR / YEAR ... 
2. DATE_SUB() : 날짜에서 원하는 만큼 빼줄 수 있음
3. MONTHNAME(3) : March : 숫자로 입력된 월을 영어로 바꿔줌
4. DAYOFMONTH('2022-03-28') : 28 

## 집계
sum / max / min / avg / count

## 변수 선언
set @IDX:=21

## 정규표현식
1. SELECT REGEXP('.') : 문자 한개로 이루어져 있는 것
2. SELECT REGEXP('...') : 문자 3개로 이루어져 있는 것<BR>


|ID|NAME|
|---|---|
|1|도레미|
|2|미파솔|
|3|파솔라|

위와 같은 테이블이 주어 진다면,
3. SELECT ID FROM TABLE WHERE NAME REGEXP('레|라') : 1,3이 출력

4. REGEXP('^BLAH') BLAH로 시작되는 단어를 찾음 (BLAHABC)
5. REGEXP('BLAH$') BLAH로 끝나는 단어를 찾음 (ABCBLAH) 
6. REGEXP('A*') A가 0회 이상 : (A,AA,BB)
7. REGEXP('A+') A가 1회 이상 : (A,AA,BBA) 
8. REGEXP('[A-z]+') : 알파벳이 들어있는 모든 문자열을 찾음
9. REGEXP('[0-9]+') : 숫자가 들어있는 모든 문자열을 찾음
10. REGEXP('&[0-9]+') : 숫자로 시작하고 숫자가 들어있는 모든 문자열을 찾음
11. REGEXP('^[0-9]') : 숫자가 들어있는 모든 문자열은 제거


