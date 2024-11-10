---
title: "Nestjs Exception / validation"
date: 2023-04-10T15:11:33+09:00
author: 홍순엽
categories:
    - nest.js
tags:
    - nest.js
    - exception
    # - authentication
    # - authorization
draft: false
featuredImage: /images/exception.jpg
hiddenFromHomePage : true
---

## Exception

Nest는 application 내에서 발생하는 에러를 핸들링하기 위해 `exception filter` 를 가지고 에러를 처리합니다. default exception filter는 `BaseExceptionFilter` 이고 다음과 같은 코드로 이루어져 있습니다. 
<br>
```javascript
// nest/packages/core/exceptions/base-exception-filter.ts

export class BaseExceptionFilter<T = any> implements ExceptionFilter<T> {
  // ...
  catch(exception: T, host: ArgumentsHost) {
    // ...
    if (!(exception instanceof HttpException)) {
      return this.handleUnknownError(exception, host, applicationRef);
    }
    const res = exception.getResponse();
    const message = isObject(res)
      ? res
      : {
          statusCode: exception.getStatus(),
          message: res,
        };
    // ...
  }
 
  public handleUnknownError(
    exception: T,
    host: ArgumentsHost,
    applicationRef: AbstractHttpAdapter | HttpServer,
  ) {
    const body = {
      statusCode: HttpStatus.INTERNAL_SERVER_ERROR,
      message: MESSAGES.UNKNOWN_EXCEPTION_MESSAGE,
    };
    // ...
  }
}
```

nest는 우리가 어플리케이션 내에서 `HttpException`으로 예외를 처리하기를 기대합니다. 그렇지 않다면, nest 는 Internal server error를 내주게 됩니다. 
<br>
`HttpException` 은 3개의 인자를 갖고 있습니다.
- `statusCode`
- `message` : string / Record(json)
- `options`

`message`에 객체를 넘겨주어 에러를 처리한다면, 이를 serialize 하여 어떠한 format으로 변경 후 response로 넘겨주게 됩니다.
<br>
options 객체에 `{cause:error}`와 같이 넘겨 준다면, 로깅 목적으로 사용할 수 있습니다.
<br>
## Exception Filter Customizing
에러를 처리할 때는 BaseExceptionFilter 를 default filter로 사용합니다. 이를 변경하고 싶다면, `catch()` 메서드 내부에 원하는 동작을 처리하면 됩니다.

```javascript
import { Catch, ArgumentsHost } from '@nestjs/common';
import { BaseExceptionFilter } from '@nestjs/core';
 
@Catch()
export class ExceptionsLoggerFilter extends BaseExceptionFilter {
  catch(exception: unknown, host: ArgumentsHost) {
    console.log('Exception thrown', exception);
    super.catch(exception, host);
  }
}
```

여기서 `@Catch()` 데코레이터 내부에 특정한 exception만 처리하도록 할 수 있습니다.
<br>
BaseExceptionFilter를 상속받게 되면 `catch()` 메서드를 수행하는데, 이 메서드에는 2개의 인자가 들어가게 됩니다.
- `HttpException` : 현재 처리 되는 예외 객체를 의미합니다.
- `AugumentsHost` : `Excecution context` 와 관련있는 객체인데, 이에 대한 설명은 길어질 것 같으므로 다음 포스팅으로 미루도록 하겠습니다.
<br>

이렇게 filter를 customizing 하고 나서는, 3가지 방법을 통해 어플리케이션에 적용할 수 있습니다.

1. main.ts에 app.useGlobalFilters 미들웨어를 등록
```

import { HttpAdapterHost, NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import * as cookieParser from 'cookie-parser';
import { ExceptionsLoggerFilter } from './utils/exceptionsLogger.filter';
 
async function bootstrap() {
  const app = await NestFactory.create(AppModule);
 
  const { httpAdapter } = app.get(HttpAdapterHost);
  app.useGlobalFilters(new ExceptionsLoggerFilter(httpAdapter));
 
  app.use(cookieParser());
  await app.listen(3000);
}
bootstrap();
```
<br>
2. AppModule 에 inject 

```
import { Module } from '@nestjs/common';
import { ExceptionsLoggerFilter } from './utils/exceptionsLogger.filter';
import { APP_FILTER } from '@nestjs/core';
 
@Module({
  // ...
  providers: [
    {
      provide: APP_FILTER,
      useClass: ExceptionsLoggerFilter,
    },
  ],
})
export class AppModule {}
```

<br>
3. `@UseFilters` 데코레이터를 사용

```
@Get(':id')
@UseFilters(ExceptionsLoggerFilter)
getPostById(@Param('id') id: string) {
  return this.postsService.getPostById(Number(id));
}
```

## Validation

