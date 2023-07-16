# Serializing Response with Interceptor


## Interceptor

@Injectable decorator로 표기되고, `NestInterceptor` interface를 구현한 클래스를 말합니다. 

AOP(Aspect Objet Programming)의 원칙에 의해 만들어진 클래스입니다. 

<br>

Interceptor는 `Intercept()` 라는 메서드를 수행해야하고, `Intercept()` 메서드에는 2개의 인자가 들어갑니다.

- `Execution Context` : `ArgumentsHost`를 상속받은 객체이고, route handler function에 전달된 argument에 접근할 수 있습니다.
- `Call Handler` : request/response를 wrapping 하고, CallHandler는 PointCut이라고 불리는 `handle()` 메서드를 수행하는데, route handler method에 도달할 지 말지 결정할 수 있습니다. `handle()` 메서드는 Rxjs의 `Observable` 객체를 리턴시키고 이 객체를 이용해서 route handler 가 종료되고 나서도 response 객체를 변경시킬 수 있습니다.

## 기능

1. method 실행 전 후에 추가적인 로직을 구성 : `tap()` operator를 사용
```javascript
@Injectable()
export class LoggingInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    console.log('Before...');

    const now = Date.now();
    return next
      .handle()
      .pipe(
        tap(() => console.log(`After... ${Date.now() - now}ms`)),
      );
  }
}
```

2. 반환될 값을 변경 : `map()` operator를 사용
```javascript
import { Injectable, NestInterceptor, ExecutionContext, CallHandler } from '@nestjs/common';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

export interface Response<T> {
  data: T;
}

// 특정 객체로의 변환
@Injectable()
export class TransformInterceptor<T> implements NestInterceptor<T, Response<T>> {
  intercept(context: ExecutionContext, next: CallHandler): Observable<Response<T>> {
    return next.handle().pipe(map(data => ({ data })));
  }
}

// null값에 대한 처리
@Injectable()
export class ExcludeNullInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    return next
      .handle()
      .pipe(map(value => value === null ? '' : value ));
  }
}

```

3. 예외를 처리하는 방법 : `catchError()` operator를 사용
```javascript
@Injectable()
export class ErrorsInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    return next
      .handle()
      .pipe(
        catchError(err => throwError(() => new BadGatewayException())),
      );
  }
}
```

4. 캐싱
```javascript
import { Injectable, NestInterceptor, ExecutionContext, CallHandler } from '@nestjs/common';
import { Observable, of } from 'rxjs';

@Injectable()
export class CacheInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const isCached = true;
    if (isCached) {
      return of([]);
    }
    return next.handle();
  }
}
```

## Binding
1. controller/ route handler에 `@UseInterceptors()`
2. main.ts 에 `app.useGlobalInterceptors()`
3. module.provider로 다음 객체를 등록
```
{
    provide:APP_INTERCEPTOR,
    useClass:SomeInterceptor
}
```

## Serializing Response

Serialization 은 user에게 response data를 보내기 전 변경하는 작업을 의미합니다.

<br>

예를 들어, 유저의 정보를 조회할 때 유저의 비밀번호는 노출시키지 않고 싶다고 한다면,` class-transformer` 의 `@Exclude()` 데코레이터를 property에 추가하기만 하면 됩니다. 그리고 나서, 컨트롤러 단에 `@UseInterceptors(ClassSerializeInterceptor)` 를 달아주거나, main.ts에 `app.useGlobalInterceptors(new ClassSerializeInterceptor(app.get(Reflect)))` 전역으로 등록해주면 됩니다.

<br>

또는, `@SerialzeOptions({strategy : "excludeAll"})` 을 데코레이터로 사용해주고, 노출시키고자 하는 속성만, `@Expose()` 를 사용해주면 됩니다.

<br>

```javascript
@Transform(value=>{
    if(value!==null) return value
})
```
와 같이 써주게 되면, null 값은 보여지지 않게 됩니다.

<br>


