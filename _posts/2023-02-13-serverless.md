---
layout: post
title: Microservice with Serverless
author: soonyubing
description: serverless, typescript 
featuredImage: null
img: null
tags: serverless nodejs typescript middy aws-lambda
categories: serverless nodejs typescript middy aws-lambda
date: '2023-02-13 11:17:00 +0900'
# image:
#   src: /assets/img/Chirpy-title-image.jpg
#   width: 1000   # in pixels
#   height: 400   # in pixels
#   alt: image alternative text
---

## 1. Serverless Framework, typescript 설치

`npm install -g serverless`

serverless 모듈을 설치하는 이유는 serverless 명령어를 사용하기 위함이다. 모듈을 설치한 다음 serverless (또는 sls) 명령어를 입력하면 새로운 프로젝트를 생성 &rightarrow; AWS Credential &rightarrow; Serverless Dashboard를 사용할 것인지를 체크한 다음 새로 생성한 프로젝트에 `serverless.yml` 파일을 생성한다. 

serverless 란 사용자가 백엔드 환경 구성에 대한 고민을 하기 보단 어플리케이션에 집중할 수 있는 환경을 구성하기 위함이다. 

`serverless.yml` 파일은 간단한 handler와 provider를 포함하고 있다. 

`sls plugin install --name serverless-offline` <br>
local 환경에서 api gateway 를 lambda 앞에 구성해주는것과 같은 효과를 누리기 위해 설치했다.

`sls plugin install --name serverless-plugin-typescript`<br>
serverless framework가 typescript 문법을 사용할 수 있도록 지원해주는 모듈을 설치했다. 

`npm i typescript ts-node`
ts-node는 nodejs의 런타임 환경에서 typescript 문법을 javascript로 변환해주고 실행시켜주는 모듈이다.

## 2. typescript는 무엇인가?
<br>
자세한 내용은 [여기](https://yozm.wishket.com/magazine/detail/1376/ "typescript")를 참고 <br>

JavaScript &rightarrow; type에 대한 체크가 없어 실행 시 빠르게 실행되지만 코드가 많아질수록 에러를 체크하기 어려움

TypeScript &rightarrow; JavaScript를 만들어내는 도구; type을 체크해서 런타임 이전에 에러를 확인할 수 있는 장점이 있으나 실행시 느려짐, 하지만 이를 위해 컴파일 전에만 타입을 체크하고 런타임시에는 javascript처럼 동작하도록 하는 기능도 개발이 되었음; JavaScript의 prototype 객체 기반 함수형 동적 타입 스크립트 처럼 개발하되 Type check 와 Auto Complete기능을 지원한다고 생각하면 되겠다.

## 3. tsyringe
<br>
Dependency Injection : tsyringe 모듈을 설명하기 전에 DI에 대해 설명하자면, DI는 한마디로, dependencies를 생성하거나 requiring 하기 보다는, parameter로 전달하는 것을 의미한다. 

tsyringe 모듈을 사용하는 이유는 DI를 제공하기 때문인데, 해당 모듈을 사용하면 각각의 레이어를 독립적으로 구성할 수 있고, 독립적으로 구성할 수 있다면 테스트하기가 쉬워진다. 

``` javascript
//users-service.js
const User = require('./User');
const UsersRepository = require('./users-repository');

async function getUsers() {
  return UsersRepository.findAll();
}

async function addUser(userData) {
  const user = new User(userData);
  
  return UsersRepository.addUser(user);
}

module.exports = {
  getUsers,
  addUser
}
```

해당 코드는 user service를 제공하는 간단한 코드이다. 해당 코드를 보면 함수 내부에서 특정 repository와 연결되어 있다. 만약 특정 repository를 다른 repository로 변경한다면, 위 코드는 모두 바꿔야 한다는 단점이 있다.

추가로, 하나의 테스트 할 때도 불편하다. 현재 코드에서 테스트를 하기 위해선 UserRepository에 대한 가짜 객체를 sinon, Jest-mock 이라는 외부 라이브러리를 사용해서 만들어야 한다. 

다음은 의존성을 주입한 코드이다. 의존성을 주입한다는 것은 함수 내에서 필요한 dependency를 생성하거나 불러오는게 아니라 인자로서 전달하는 것이다.

```javascript
const User = require('./User');

function UsersService(usersRepository) { // check here
  async function getUsers() {
    return usersRepository.findAll();
  }
  
  async function addUser(userData) {
    const user = new User(userData);
  
    return usersRepository.addUser(user);
  }
  
  return {
    getUsers,
    addUser
  };
}

module.exports = UsersService
```

이렇게 코드를 구성한다면, 테스트 코드를 다음과 같이 구성할 수 있다.

``` javascript
const UsersService = require('./users');
const assert = require('assert');

describe('Users service', () => {
  it('gets users', async () => {
    const users = [{
      id: 1,
      firstname: 'Joe',
      lastname: 'Doe'
    }];
    
    const usersRepository = {
      findAll: async () => {
        return users
      }
    };
    
    const usersService = new UsersService(usersRepository);
    
    assert.deepEqual(await usersService.getUsers(), users);
  });
});
```

