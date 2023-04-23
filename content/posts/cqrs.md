---
title: "쿼리 성능 올리는 방법 1: CQRS"
subtitle: ""
date: 2023-04-24T02:12:03+09:00
lastmod: 2023-04-24T02:12:03+09:00
draft: false
author: "홍순엽"
authorLink: ""
description: ""
license: ""
images: []

tags: ["cqrs","nestjs"]
categories: ["backend","nest.js"]

featuredImage: "images/cqrs_feature.jpg"
featuredImagePreview: "images/cqrs_feature.jpg"

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
CQRS는 사실 DB에 많은 요청이 있는 서비스를 운영해보지 않으면 경험할 수 없을 거라 생각한다. 나 또한 이번에 배우면서, 나중에 DB 성능을 높이기 위해 이런 패턴도 있었지 하면서 떠올리기 위해 포스트를 작성한다. <br>

## CQRS

### what?
읽기와 쓰기(업데이트) 작업을 분리한 디자인 패턴을 의미한다.

### why?
하나의 DB만 운영하게 되면 겪는 비효율적인 join을 제거하고, 복잡한 쿼리를 피할 수 있다. DB에 DB서버가 처리할 수 있는 요청보다 많은 작업이 들어오게 되면 lock이 걸리게 되고, 이는 다른 작업을 처리하지 못함을 의미한다. <br>
그렇기 때문에, 읽기용 DB와 쓰기용 DB를 따로 배치하고 두 DB를 동기화시키면 위에서 일어나는 문제를 해결할 수 있다. 

### how?
쿼리용 데이터모델에는 join이나 복잡한 쿼리에 대응할 수 있는 데이터를 미리 가공하여 준비하는 방식으로 해결할 수 도 있다.
<br>
읽기용 DB와 쓰기용 DB를 서로다른 성격의 DB를 사용해서 운영할 수 있다. 예를 들어 인스타그램에서는 스토리 READ를 위해 NOSQL인 Cassandra를 이용하고, 사용자 정보 저장에는 RDB인 Postgre를 사용하는 것처럼 말이다.

### additional
DB를 분리시켰으면 중요한 것은 두 DB가 동기화 되어야 한다는 점이다. 사용자의 정보를 저장하고 두 DB를 연결하는 메세지 큐가 처리하기 전에 사용자 정보를 조회할 때 잘못된 쿼리결과를 받아서는 안되기 때문이다. 

## 구현

다음으로는 Nest를 이용해 CQRS를 구현해보겠다. Nest에서는 `@nestjs/cqrs` 의 `CommandBus`, `QueryBus` 를 이용해 특정 명령을 실행시킬 수 있다. 이 명령을 실행시키게 되면, 각각 `@CommandHandler()` / `@QueryHandler()` 로 감싸준 클래스에서 내가 원하는 동작을 구현해주면 되겠다. 이때 클래스는 `ICommandHandler` / `IQueryHandler` interface를 구현한 것으로 **`execute`** 메서드를 반드시 실행시켜주어야 한다.

<br>

Comment 서비스를 CQRS로 구현한다면 다음과 같은 구조를 가지게 될 것이다.

{{< figure src="/images/20230424_024627.png" title="Comments Logic Structure" >}}

### command
command는 다음과 같이 DB작업에 쓰일 것들을 생성자에 집어넣어서 생성해준다. <br>
```javascript
export class CreateCommentCommand {
  constructor(
    public readonly comment: CreateCommentDto,
    public readonly author: User,
  ) {}
}

export class GetCommentsQuery {
    constructor(
      public readonly postId?: number,
    ) {}
  }
```

### handler & MODEL
Handler는 Command 클래스에 들어있는 생성자 값을 읽고, `execute` 내부에서 호출하여 사용할 수 있다.

<br>

```javascript
import { CommandHandler, ICommandHandler } from "@nestjs/cqrs";
import { CreateCommentCommand } from "../implementations/createComment.command";
import { InjectRepository } from "@nestjs/typeorm";
import Comment from "src/comments/entities/comment.entity";
import { Repository } from "typeorm";


@CommandHandler(CreateCommentCommand)
export class CreateCommentHandler implements ICommandHandler<CreateCommentCommand>{
    constructor(
        @InjectRepository(Comment) private commentsRepository : Repository<Comment>
    ){}
    
    async execute(command: CreateCommentCommand): Promise<any> {
        const newPost = await this.commentsRepository.create({
            ...command.comment, // content , post 
            author : command.author
        });

        await this.commentsRepository.save(newPost);

        return newPost;
    }
}


import { InjectRepository } from "@nestjs/typeorm";
import { GetCommentsQuery } from "../implementations/getComments.query";
import {IQueryHandler, QueryHandler} from "@nestjs/cqrs";
import Comment from "src/comments/entities/comment.entity";
import { Repository } from "typeorm";

@QueryHandler(GetCommentsQuery)
export class GetCommentsHandler implements IQueryHandler<GetCommentsQuery>{
    constructor(
        @InjectRepository(Comment) private commentsRepository : Repository<Comment>
    ){}

    async execute(query: GetCommentsQuery): Promise<any> {
        if(query.postId){
            return this.commentsRepository.findOne({where:{id:query.postId}});
        }
        return this.commentsRepository.find();
    }
}
```

전체코드는 [여기](https://github.com/soonyubi/nestjs-typescript/tree/master/src/comments)에서 확인할 수 있다.

### 마무리
항상 모든 서비스를 완전한 상태로 개발한다는 것은 쉽지 않다. 서비스가 장애에 부딪히고, 성능이 예상만큼 안따라주면 왜 그럴까 고민해가는 과정에서 성장해나간다고 생각한다. 그런 관점에서 나도 디자인 패턴을 알기만 하는게 아니라, 실제 그런 상황을 겪고 이러한 방법을 떠올려서 해결해보는 과정을 경험해보고 싶다.