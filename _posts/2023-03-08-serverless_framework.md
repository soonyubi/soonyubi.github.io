---
layout: post
title: Serverless Framework 를 이용해 프로젝트 준비하기
author: soonyubing
description: serverless framework 프로젝트를 준비하기 앞서 공부한 내용을 까먹을까봐 미리 정리해둔 포스트 
featuredImage: null
img: null
tags: [serverless lambda]
categories: [backend nodejs]
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

## 개요
해당 포스트는 serverless framework를 사용해 프로젝트를 구성하기 앞서 유튜브 강의를 통해 새롭게 알게 된 내용과 까먹지 않기 위해 정리한 포스트 입니다.

---




## Prerequisite

```shell
npm i -g serverless
```
위 명령어를 이용해 serverless 를 전역으로 설치합니다. 설치한 이후엔 serverless 또는 sls 명령어를 사용할 수 있습니다. <br>

serverless 명령어를 입력하게 되면 serverless framework 의 새로운 프로젝트를 생성하는데, 많은 선택 조건이 있습니다..

1. `AWS Node.js HTTP API` : API Gateway 와 Lambda 를 사용해서 Node.js로 작성된 HTTP API를 생성합니다. REST 또는 GraphQL로 구성할 수 있습니다.
2. `AWS Node.js Express API` : Express 를 사용해서 API를 생성합니다. Express를 사용하기 때문에 다양한 미들웨어를 사용해 기능을 확장할 수 있습니다.
3. `AWS Node.js Scheduled Function` : Lambda 와 CloudWatch Event를 사용하여 주기적으로 실행되는 작업을 구현할 수 있습니다. 이를 통해 스케쥴링 기능을 갖춘 서버리스 어플리케이션을 개발할 수 있습니다.
4. `AWS Node.js WebSocket API` : 실시간 데이터 스트리밍 
5. `AWS Node.js EventBridge` : 비동기 이벤트 처리
6. `AWS Node.js SQS Worker` : SQS 메세지 워커를 생성
7. `AWS Node.js S3 Event` : S3 버킷이벤트 처리기를 생성 

이 외에도 python 을 이용하거나 다른 서비스를 생성하는 많은 프로젝트를 생성할 수 있고, 자세한 설명은 해당 프로젝트를 하게 되면 설명하는 것으로 하고 넘어가겠습니다.

```shell
serverless plugin install --name serverless-plugin-typescript
```
위 명령어는 다음과 같은 역할을 합니다. Typescript로 개발된 코드를 Javascript로 컴파일을 하고, AWS Lambda 함수를 배포하기 위한 CloudFormation 스택을 생성합니다.

```shell
npm install --dev typescript ts-node
npm install aws-lambda
npm install --dev @types/aws-lambda
```
`ts-node`는 typescript 코드를 javascript로 컴파일하지 않고, 바로 실행하고 디버깅하는 용도로 사용하기 위해 설치합니다.

```shell
tsc --init
```
typescript 프로젝트를 초기화하기 위해 사용하는데, 해당 명령어를 사용하게 되면 `tsconfig.json` 파일이 생깁니다. tsconfig.json은 typescript 코드를 javascript로 컴파일할 때 컴파일러의 옵션을 정의해줄 수 있습니다.

현재 프로젝트 내에서는 다음과 같이 구성되어 있습니다.
```javascript
{
  "compilerOptions": {
    "module": "CommonJS",
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "target": "ES6",
    "resolveJsonModule": true,
    "noImplicitAny": true,
    "moduleResolution": "node",
    "sourceMap": true,
    "baseUrl": ".",
    "emitDecoratorMetadata": true,
    "experimentalDecorators": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true
  },
  "include": ["app/**/*", "*.ts"],
  "exclude": ["node_modules"]
}
```
- "module": CommonJS : <br>CommonJS 모듈 시스템을 사용합니다. CommonJS는 Node.js에서 사용되는 모듈 시스템으로, require() 함수를 사용하여 모듈을 로드합니다.
- "esModuleInterop": true <br> ES 모듈과 CommonJS 모듈 간의 상호운용성을 제공하는 옵션입니다. 이를 true로 설정하면 import 문을 사용하여 CommonJS 모듈을 로드할 수 있습니다.
- "allowSyntheticDefaultImports": true <br> 모듈의 디폴트 내보내기(default export)를 가질 수 있는 모듈을 CommonJS로부터 가져올 때 합성된 디폴트 내보내기를 허용합니다.
- "target": ES6 <br> 컴파일된 JavaScript 코드의 ECMAScript 버전을 지정합니다. 이 경우 ES6(ES2015)을 지정하여 ES6에서 새로 도입된 문법과 기능을 사용할 수 있습니다.
- "resolveJsonModule": true <br> .json 파일을 모듈로 가져올 수 있게 해주는 옵션입니다.
- "noImplicitAny": true <br> 암시적 any를 허용하지 않습니다. 이 설정을 사용하면 함수의 매개변수나 반환값의 타입이 명시되지 않은 경우 에러를 발생시킵니다.
- "moduleResolution": node <br> 모듈 해석 방식을 지정합니다. 이 경우 Node.js의 모듈 해석 알고리즘을 사용합니다.
- "sourceMap": true <br> 컴파일된 JavaScript 코드와 TypeScript 소스 코드 간의 매핑 정보를 생성합니다. 이를 통해 디버깅이 용이해집니다.
- "baseUrl": "." <br> 상대 경로 모듈 해결에 사용할 기본 경로를 지정합니다. 이 경우 프로젝트 루트를 기준으로 설정합니다.
- "emitDecoratorMetadata": true <br> 데코레이터 사용을 위해 추가적인 메타데이터를 생성합니다.
- "experimentalDecorators": true <br> 실험적인 데코레이터 문법을 사용할 수 있게 해주는 옵션입니다.
- "forceConsistentCasingInFileNames": true <br> 파일 이름이 대/소문자 구분 규칙에 맞게 작성되었는지 검사합니다.
- "skipLibCheck": true <br> 라이브러리 파일을 체크하지 않습니다. 이를 true로 설정하면 컴파일 속도를 높일 수 있습니다.
- "include": ["app/**/", ".ts"] <br> 컴파일 대상 파일을 지정합니다. 

---

## API 생성

일단 serverless yml에 함수를 정의해야 합니다. 함수를 정의하게 되면 이벤트 트리거를 연결하고 필요한 리소스를 구성할 수 있습니다. 

```javascript
functions:
  signup:
    handler: app/handler.Signup
    events:
      - httpApi:
          path: /signup
          method: get
```

현재 저는 signup 이벤트에 대한 함수는 app/handler에 작성했습니다. 

app/handelr.ts 의 내용은 다음과 같습니다.
```javascript
export const Signup = (event : APIGatewayProxyEventV2) =>{
  // some bussiness logig
  // return response
}
```

여기서 parameter로 들어오는 `APIGatewayProxyEventV2` 에는 여러 정보가 포함되어 있습니다. header, body, cookies 등 많은 정보가 포함되어 있습니다.

이 때 body에 들어온 json 객체를 parsing 하기 위해 `middy` 에서 제공하는 미들웨어를 설치하고자 합니다. 
`middy`는 node.js 에서 aws-lambda를 사용할 때 사용할 수 있는 미들웨어 패키지입니다. 현재 저희 목표는 json 을 parsing 하는 것이므로 다음과 같은 명령어를 설치한 후 app/handler.ts 파일을 다음과 같이 수정합니다.

```shell
yarn add @middy/http-json-body-parser
```

```javascript

import bodyParser from "@middy/http-json-body-parser";

export const Signup = middy((event : APIGatewayProxyEventV2) =>{
  // some bussiness logig
  // return response
}).use(bodyParser());
```

handler 에서 모든 비지니스로직을 처리하고, 데이터베이스에 연결을 요청하고 데이터를 fetch 하는 작업을 모두 쓴다면 유지보수가 힘들기 때문에, service/userService.ts repository/userRepository.ts 로 구분하여 작성하도록 하겠습니다. 

이때, userService에서는 repository 객체가 필요한데, userService class 내에서 새로운 repository 객체를 생성할 수 도 있지만, 나중에 테스트할 때 3rd party 모듈을 사용해 mock 객체를 생성하는 번거로움이 생깁니다. 그러므로 `tsyringe` 모듈을 사용해 userService로 userRepository에 대한 객체의 의존성을 주입하도록 하겠습니다. 

```javascript
import autoInjectable from "tsyringe"

@autoInjectable
export class userServcie {
  repository : UserRepository;
  constructor(repository : UserRepository){
    this.repository = repository;
  }
}
```

reflect-metadata 를 import 해줘야 하는데 해당 이유는 컴파일 타임에 데코레이터와 메타데이터를 전달할 수 있도록 도와주는 기능을 추가해준다고 생각하면 되겠습니다. 
(자세한 내용은 나중에 추가 예정)

---

## 배포
배포를 하기 위해선 serverless.yml 파일을 수정해야 합니다. 

```javascript
service: aws-node-http-api-project
frameworkVersion: '3'

provider:
  name: aws
  runtime: nodejs18.x
  versionFunctions: false
  stage: "dev"
  region: "ap-northeast-2"
  httpApi:
    cors: true

functions:
  signup:
    handler: app/handler.Signup
    events:
      - httpApi:
          path: /signup
          method: get
```

aws provider를 사용할 예정이고, 실행환경은 node.js 18.x 입니다. lambda 함수의 versioning은 false로 하고 현재 배포 단계는 dev이고, 서울 리전에 배포합니다. `httpApi`  필드는 주로 API Gateway를 구성하는 데 사용합니다. 현재 `cors : true`로 세팅 되어 있으며, cross-origin request를 허용한다는 의미 입니다.

배포 과정에서 다음과 같은 에러가 발생할 수 있습니다. 
```shell
npm Error : EBUSY resource busy or locked
```
이럴경우 해결책은 다음과 같습니다. 
1. [해결법 링크](https://github.com/npm/npm/issues/13461)
2. vscode를 종료시키고 관리자 terminal에서 배포해봅니다.