# Dependency Injection (feat:Nest)


## 개요
Nest.js Framework(Spring, Angular 등)를 사용해서 작업을 하다보면, Dependency Injection, Inversion of Controll 같은 단어들이 자주보인다. 필자는 이 단어들이 어떤 의미를 갖는지 모르고 개발을 하다보니 계속 구렁텅이에 빠지는 기분이라 이에 대해 많은 포스트/ 비디오를 보고 정리하였다.
<br><br>
포스트는 다음과 같은 순서로 구성된다. Nest.js에서 DI를 어떻게 사용하는지를 Nest.js 메인테이너의 설명을 토대로 작성하였다. 그리고 DI가 개념적으로 무엇인지, 왜 사용해야 하는지, 어떻게 사용하면 되는지에 대해 포스트하려고 한다.

## Nest.js 에서 DI란?
Nest를 사용하다보면 다음과 같은 코드를 자주 볼 수 있다.
```javascript
@Injectable()
class CatService{
    constructor(
        private readonly httpService : HttpService,
        private readonly logger : Logger,
    ){}
}
```
여기서 `@Injectable()` 이란 데코레이터에 주목해야 한다. 이 데코레이터는 typescript compiler에 의해 다음과 같은 metadata를 생성한다. 
<br><br>

```javascript
CatsService = __decorate(
    [__metadata("design:paramtypes",[HttpService, Logger])],
    CatsService
)
// metadata  = [HttpService, Logger]
```
<br>

여기서 `"design:paramtypes"` 는 typescript metadata key이고, 이를 이용해 CatsService가 참조하는 class 배열을 얻을 수 있다.
이러한 metadata는 Metadata Reflection API를 이용해 다음의 함수를 실행하여 
```javascript
Reflect.getMetadata("design:paramtypes",CatsService)
```
<br>
현재 CatsService가 참조하는 의존성이 어떤 타입인지 알 수 있게 된다. 다음 그림과 같이 `INJECTOR` 는 해당 타입에 해당하는 Instance를 `DI Controller`에게 요청하고 `DI Controller`는 해당 타입에 맞는 적절한 intance를 `INJECTOR`에게 반환해준다. `INJECTOR`는 해당 값을 사용해서 새로운 provider를 인스턴스화함으로써 DI를 수행한다. <br>
{{< admonition tip "" >}}
실제로 내부적으로는 Circular Dependency, Synchronous Process 등과 같은 복잡한 것들도 처리하지만, 이 글의 내용에서 벗어나는 내용이므로 다른 포스트에 올리도록 하겠다.
{{< /admonition >}}

{{< figure src="/images/20230416_220537.png" title="" >}}
<br>
### Injector in Nestjs
Nest.js에서는 module 레벨의 `Injector`만을 가지고 있습니다. 그리고, Nest.js에서는 모듈의 구조를 모듈을 정점으로 하는 그래프로 표한할 수 있습니다. 그리고 각 모듈은 Injector를 가지고 있다. <br>
Nest.js에서 각 모듈이 `singleton` 처럼 보이기는 하지만, 사실은 `Dynamic module` 이 있기 때문에 꼭 그렇지는 않는다. 모듈이 이러한 구조를 갖고 내부적으로는 `exports`/ `imports` / `declarations` 을 통해 동적으로 확장할 수 있다. 이러한 기능을 제공하는 이유는 모듈의 isolation을 위함이다. <br><br>
A module에서 B module의 B service를 이용하고 싶다면, A module에 B moudle을 import하고, B Service를 B module에서 `exports` 시켜야 한다. 만약 B service를 `exports` 하지 않으면, 캡슐화되어 다른 모듈에서 접근할 수 없게 된다. 이 점이 Nest.js 의 DI에서 중요한 점이다.

Nest는 다음과 같은 순서로 의존성을 해결합니다.
1. 현재 module 내에서 provider를 찾는다. 
2. import된 module을 확인한다. 
3. 만약 모듈이 exports 되지 않는다면 해당 provider를 사용할 수 없다.

이렇게 Nest.JS는 격리수준에 있어서 엄격하게 작용합니다. 그리고 해당 provider를 찾을 때 그래프 상에서 현재 ENQUIRER에 가까운 provider를 사용하게 된다.

{{< figure src="/images/20230416_222342.png" title="" >}}

### Global Scope
{{< figure src="/images/20230416_222433.png" title="" >}}
만약 core module을 `@Global()` 데코레이터를 사용해서 전역으로 설정하면, 다른 module에서 가상의 간선을 해당 모듈에 연결하게 된다. 그렇게 되면, core module을 사용하고자 할 때, 명시적으로 imports 하지 않고 해당 모듈을 사용할 수 있다.  


## DI란?
> 상위 모듈은 하위 모듈에 의존해선 안된다. <br>
추상화된 것은 구체화된 것을 의존하면 안된다, 구체화된 것은 추상화된 것을 의존한다.
## 왜 DI를 사용해야하고 어떻게 사용하는지?
button class가 있다고 가정하자. button class는 `누른다` 라는 기능을 수행한다. 누르는 행위로 on/off 되는 객체를 생각해보자. Lamp / Carkey 가 있다고 가정하자. 이 때 button을 눌렀을 때 lamp는 불이 켜지다 / 불이 꺼지다의 기능을 수행해야 하고, Carkey는 문을 열다/ 문을 닫는다의 기능을 수행해야 한다. <br>
근데 button class가 lamp에도 의존하고, Carkey에도 의존하게 된다면, lamp, carkey 객체를 button에 생성해야하고, 각각의 기능도 따로 구현해야 한다. 이렇게 되면 button class는 재사용함의 가치를 상실하게 된다. 
<br><br>
이렇게 하지 않으려면, button을 추상화 시켜야 한다. ButtonInterface라고 한다면, button은 buttonInterface를 의존하고, lamp, carkey 도 buttonInterface를 의존한다. 
그리고 lamp, carkey 클래스에서, ButtonInterface의 메서드인 `누른다`의 기능을 각각 정의하게 되면, Button class는 이제 lamp, carkey를 의존하지 않아도 된다. 

다른 목적으로는, 테스트에 용이하단 장점도 있다. mock 객체를 생성하거나, 3rd 라이브러리를 사용하지 않아도 된다.

## 마무리
내 포스트에 틀린점이 있다면, 필자의 부족함 때문이다. 영어로 된 유튜브를 보고 30분짜리 영상을 모두 해석해서 읽고, 몇몇 한국어 블로그 중 믿을 만한곳을 참고해서 적으려고 노력했다. 
<br>



