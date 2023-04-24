---
title: "Nest.js - Cache"
subtitle: ""
date: 2023-04-24T20:19:55+09:00
lastmod: 2023-04-24T20:19:55+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: ["nest.js","cache"]
categories: ["nest.js"]

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

## 개요
데이터베이스에서 데이터를 조회하는 요청 과정을 살펴보면, 다음과 같은 순서로 진행된다. <br>
`middleware`-`guard`-`interceptor`-`endpoint`-`service layer`-`database layer` <br>

데이터베이스까지 가는 과정은 상당히 긴 편이다(사실 pipe, filter 까지 포함하면 더 길다). 똑같은 데이터를 조회하는데 이러한 과정을 계속 반복하는 것은 불필요한 일이다. 

이번 포스트에서는 `interceptor` layer에서 캐시를 이용해 최근에 조회된 데이터를 더 빠르게 응답하는 방법에 대해 알아보고자 한다.

## 1. 내부 라이브러리를 이용한 caching

### CacheInterceptor

```javascript
@Injectable()
export class CacheInterceptor implements NestInterceptor {
  protected allowedMethods = ['GET'];
  constructor(
    @Inject(CACHE_MANAGER) protected readonly cacheManager: any,
    @Inject(REFLECTOR) protected readonly reflector: any,
  ) {}

  async intercept(
    context: ExecutionContext,
    next: CallHandler,
  ): Promise<Observable<any>> {
    const key = this.trackBy(context); // 캐시 key를 구성, 캐시할 수 없는 경우 falsy값을 반환
    if (!key) return next.handle(); // 캐시할 수 없는 경우 비즈니스로직을 처리

    try {
      const value = await this.cacheManager.get(key); // 캐시된 데이터를 확인
      if (!isNil(value)) return of(value); // 캐시된 데이터가 있는 경우 캐시된 데이터를 응답

      // 캐시된 데이터가 없는 경우
      // @CahceTTL() 데코레이터로 작성한 캐시 TTL(Time To Live) 메타데이터 조회
      const ttlValueOrFactory = this.reflector.get(CACHE_TTL_METADATA, context.getHandler()) ?? null;
      const ttl = isFunction(ttlValueOrFactory)  ? await ttlValueOrFactory(context) : ttlValueOrFactory;
      return next.handle().pipe(
        tap(response => {
          // 응답하며 동시에, 응답 데이터 캐싱 처리 진행(입력한 TTL 만큼)
          const args = isNil(ttl) ? [key, response] : [key, response, { ttl }];
          this.cacheManager.set(...args);
        }),
      );
    } catch {
      return next.handle(); // 캐시 처리, 캐시 조회 과정에서 오류 발생의 경우 비즈니스로직을 그대로 처리
    }
  }
}
```

CacheInterceptor는 어떤 컨트롤러에서 cache를 사용하게 하고 싶을때, interceptor가 해당 cache 값을 리턴할 수 있을지 없을지 판단한다. 

위 코드는 다음의 내용을 포함한다. 

- `trackBy` 메서드를 통해 캐시처리할 수 있는 요청인지 확인한다. 기본적으로 GET 엔드포인트에 대해서만 처리하도록 구성되어 있다.

- `@CacheKey()` 데코레이터로 캐시 값을 설정하지 않았다면, default cache 값은 요청 url이다. trackBy를 cacheKey를 이용하도록 overriding 할 수 있다. 해당 예제는 아래에서.

- cache key가 존재하는 경우, cache manager를 통해 해당 cache key를 가진 데이터가 존재하는 지 확인한다. 

- 일치하는 cache data가 없다면, `@CacheKey()`/ `@CacheTTL()` 을 통해 입력한 metadata를 조회한다. 

- 해당 cache data를 RxJs 를 이용해 처리한다. 

### CacheInterceptor Customizing

다음과 같이 내장된 `CacheInterceptor를` 이용해 캐싱을 처리한다면, `/posts/` 와 `/posts?search=title3`를 똑같이 처리할 것이다. 

```javascript
@UseInterceptors(CacheInterceptor)
@CacheKey(GET_POSTS_CACHE_KEY)
@CacheTTL(120)
@Get()
async getPosts(
  @Query('search') search: string,
  @Query() { offset, limit, startId }: PaginationParams
) {
  if (search) {
    return this.postsService.searchForPosts(search, offset, limit, startId);
  }
  return this.postsService.getAllPosts(offset, limit, startId);
}
```

따라서 우리는 `CacheInterceptor를` 확장해서 사용해야 한다. 다음 코드는 우리가 만약 @CacheKey를 통해 캐시 키를 전달하지 않는다면, CacheInterceptor가 cache를 처리하는 방식으로 동작하고, 캐시 키를 전달하였다면, 새로운 캐시키를 생성해 캐시를 생성할 것이다. (e.g POSTS_CACHE-null / POSTS_CACHE-search=hello) 

```javascript
import { CACHE_KEY_METADATA, CacheInterceptor, ExecutionContext, Injectable } from '@nestjs/common';
 
@Injectable()
export class HttpCacheInterceptor extends CacheInterceptor {
  trackBy(context: ExecutionContext): string | undefined {
    const cacheKey = this.reflector.get(
      CACHE_KEY_METADATA,
      context.getHandler(),
    );
 
    if (cacheKey) {
      const request = context.switchToHttp().getRequest();
      return `${cacheKey}-${request._parsedUrl.query}`;
    }
 
    return super.trackBy(context);
  }
}
```
> @CacheKey() : 특정 cache key를 지정하기 위한 데코레이터 <br> 
> @CacheTTL() : 해당 cache key의 TTL을 지정하기 위한 데코레이터

### 캐시 무효화 
캐시를 무한정으로 길게 가져가면 데이터를 검색하는 속도를 빠르게 할 수 있으니깐 좋은게 아닌가? 라는 생각도 들 수 있지만, 데이터는 계속 변하기 때문에, 그렇지만도 않는다. 우리는 새로운 데이터가 추가, 삭제, 업데이트 될 때 해당 cache key를 가진 cache 값을 무효화하고, 새로운 캐시 데이터를 생성해야 한다. 

캐시 무효화는 보통 서비스 layer에서 구현하게 되는데, 이는 몇가지 문제점을 가진다. 

- 캐시를 무효화하는 동일한 코드가 반복됨
- 캐시 관리는 주요 로직이 아니라 부가적인 로직에 가까움 

따라서 캐시 무효화를 위한 작업을 위와 같이 `CacheInterceptor를` 상속받아 별도의 Interceptor 내부에서 구현해 줄 수 있다. [해당 블로그 참고](https://hwasurr.io/nestjs/caching/)

```javascript
// src/core/httpcache.interceptor.ts
import {
  CacheInterceptor,
  CallHandler,
  ExecutionContext,
  Injectable,
} from '@nestjs/common';
import { Request } from 'express';
import { Cluster } from 'ioredis';
import { Observable, tap } from 'rxjs';

@Injectable()
export class HttpCacheInterceptor extends CacheInterceptor {
  private readonly CACHE_EVICT_METHODS = [
    'POST', 'PATCH', 'PUT', 'DELETE'
  ];

  async intercept(
    context: ExecutionContext,
    next: CallHandler<any>,
  ): Promise<Observable<any>> {
    const req = context.switchToHttp().getRequest<Request>();
    if (this.CACHE_EVICT_METHODS.includes(req.method)) {
      // 캐시 무효화 처리
      return next.handle().pipe(tap(() => this._clearCaches(req.originalUrl)));
    }

    // 기존 캐싱 처리
    return super.intercept(context, next);
  }

  /**
   * @param cacheKeys 삭제할 캐시 키 목록
   */
  private async _clearCaches(cacheKeys: string[]): Promise<boolean> {
    const client: Cluster = await this.cacheManager.store.getClient();
    const redisNodes = client.nodes();

    const result2 = await Promise.all(
      redisNodes.map(async (redis) => {
        const _keys = await Promise.all(
          cacheKeys.map((cacheKey) => redis.keys(`*${cacheKey}*`)),
        );
        const keys = _keys.flat();
        return Promise.all(keys.map((key) => !!this.cacheManager.del(key)));
      }),
    );
    return result2.flat().every((r) => !!r);
  }
}
```

전체 코드는 다음에서 확인할 수 있다.
[코드](https://github.com/soonyubi/nestjs-typescript/tree/master/src/posts)

## 2. Redis를 이용한 caching