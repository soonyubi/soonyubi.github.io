# Nestjs: Gaurd를 이용해 Authentication/ Authorization 구현


## big picture
1. passport와 bcrypt를 이용하여 사용자 인증
    - passport-local 을 이용해 local strategy 생성
    - local strategy 내부에서 validate() 를 호출하는데, validate 내부에서 bcrypt를 이용해 비밀번호를 암호화



## 1. passport와 bcrypt를 이용하여 사용자 인증

User module에 User Entity를 생성하고, User service에서 User database에 create / fetch 하는 메서드를 간단히 만들어 줍니다. UserService를 Authentication module에서 사용하길 원하기 때문에, UserService를 `@Injectable()` 데코레이터로 감싸주고, UserModule에서 UserService를 export해줍니다. 

password는 가장 안전해야 하는 데이터입니다. 그래서 password는 `hash` 해야 합니다. `hash`를 하는 과정에서 필요한 값은 random string 인 `salt` 값이 필요합니다. 

이 모든 과정을 bcrypt 라이브러리를 사용하면 쉽게 할 수 있습니다. bcrypt로 password에 salt값을 적용해 여러번 hash하여 복원하는 것을 어렵게 합니다. bcrypt는 cpu를 잡아먹는 작업이지만, thread pool의 추가적인 thread를 이용해 연산을 수행하므로 암호화하는 과정에서 다른 작업을 수행할 수 있습니다.

{{< admonition note "How to choose search engine?" >}}
The following is a comparison of two search engines:

{{< /admonition >}}



