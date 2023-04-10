---
title: "Nestjs: Gaurd를 이용해 Authentication/ Authorization 구현"
author: 홍순엽
date: 2023-04-05T20:50:23+09:00
categories:
    - backend
tags:
    - nest.js
    - guard
    - authentication
    - authorization
featuredImage: /images/guard.jpg
images :
draft: false
---

## big picture
- [x] passport와 bcrypt를 이용하여 `Authentication`
- [x] passport와 jwt를 이용하여 `Authorization` 


## 1. passport와 bcrypt를 이용하여 사용자 인증

User module에 User Entity를 생성하고, User service에서 User database에 create / fetch 하는 메서드를 간단히 만들어 줍니다. UserService를 Authentication module에서 사용하길 원하기 때문에, UserService를 `@Injectable()` 데코레이터로 감싸주고, UserModule에서 UserService를 export해줍니다. 

password는 가장 안전해야 하는 데이터입니다. 그래서 password는 `hash` 해야 합니다. `hash`를 하는 과정에서 필요한 값은 random string 인 `salt` 값이 필요합니다. 

{{< admonition note "bcrypt" >}}
이 모든 과정을 bcrypt 라이브러리를 사용하면 쉽게 할 수 있습니다. bcrypt로 password에 salt값을 적용해 여러번 hash하여 복원하는 것을 어렵게 합니다. bcrypt는 cpu를 잡아먹는 작업이지만, thread pool의 추가적인 thread를 이용해 연산을 수행하므로 암호화하는 과정에서 다른 작업을 수행할 수 있습니다.
{{< /admonition >}}

Authentication module을 생성하고, Authentication Service 에서 bcrypt를 이용해 요청으로 받은 비밀번호를 암호화하고 저장하겠습니다. 저장하기 위해서 user service가 필요하니 생성자에 전에 export한 user service를 불러와줍니다.
<br><br>
**authentication/authentication.service.ts**
```javascript
export class AuthenticationService {
  constructor(
    private readonly usersService: UsersService
  ) {}
 
  public async register(registrationData: RegisterDto) {
    const hashedPassword = await bcrypt.hash(registrationData.password, 10);
    try {
      const createdUser = await this.usersService.create({
        ...registrationData,
        password: hashedPassword
      });
      createdUser.password = undefined;
      return createdUser;
    } catch (error) {
      if (error?.code === PostgresErrorCode.UniqueViolation) {
        throw new HttpException('User with that email already exists', HttpStatus.BAD_REQUEST);
      }
      throw new HttpException('Something went wrong', HttpStatus.INTERNAL_SERVER_ERROR);
    }
  }
  
  
    public async getAuthenticatedUser(email: string, plainTextPassword: string) {
        try {
            const user = await this.usersService.getByEmail(email);
            await this.verifyPassword(plainTextPassword, user.password);
            user.password = undefined;
            return user;
        } catch (error) {
            throw new HttpException('Wrong credentials provided', HttpStatus.BAD_REQUEST);
        }
    }
    
    private async verifyPassword(plainTextPassword: string, hashedPassword: string) {
        const isPasswordMatching = await bcrypt.compare(
            plainTextPassword,
            hashedPassword
        );
        if (!isPasswordMatching) {
            throw new HttpException('Wrong credentials provided', HttpStatus.BAD_REQUEST);
        }
    }
}
```
<br><br>
> `createdUser.password = undefined;` 는 password를 response로 보내주기 위한 깔끔한 방법은 아닙니다. 나중에 수정하도록 하겠습니다.

위 함수에서 주목할 부분은 회원가입을 할 때는 비밀번호를 `bcrypt`의 `hash` 메서드를 이용해 hash 하고 login 할 때는 `compare` 메서드를 이용해 요청값과 DB에 저장된 비밀번호를 비교하는 것입니다.

여기까지 인증로직을 구현했으니, passport와 authentication을 통합하는 일만 남았습니다. passport는 authentication을 추상화하여 우리가 좀 더 다른 로직에 집중할 수 있게 해줍니다.
<br><br>
```
npm install @nestjs/passport passport @types/passport-local passport-local @types/express
```
<br><br>
**authentication/local.strategy.ts**
```javascript
import { Strategy } from 'passport-local';
import { PassportStrategy } from '@nestjs/passport';
import { Injectable } from '@nestjs/common';
import { AuthenticationService } from './authentication.service';
import User from '../users/user.entity';
 
@Injectable()
export class LocalStrategy extends PassportStrategy(Strategy) {
  constructor(private authenticationService: AuthenticationService) {
    super({
      usernameField: 'email'
    });
  }
  async validate(email: string, password: string): Promise<User> {
    return this.authenticationService.getAuthenticatedUser(email, password);
  }
}
```
<br><br>
{{< admonition note "how LocalStrategy works on nestjs?" >}}
NestJs는 Passport 라이브러리를 사용하여 인증을 구현하는데, LocalStrategy는 인증 전략 중 하나로 로그인 폼에서 이름/ 패스워드를 이용해 인증하는 방식을 의미합니다.

LocalStrategy를 이용하는 경우 사용자를 검증하는 방식을 정의해야 하므로 validate() 함수 내부에 구현하였습니다.

NestJs에서는 LocalStrategy를 구현하기 위해 `@nestjs/passport` 모듈을 사용합니다. 이 모듈에서 `AuthGuard` 클래스를 사용하여 인증된 요청만 허용하는 Guard를 쉽게 생성할 수 있습니다. 

`@Guard(AuthGuard('local'))`을 라우터의 미들웨어로 등록하여 쉽게 인증을 구현할 수 있습니다.
{{< /admonition >}}

LocalStrategy를 구현하고 나서는 Authentication module에 등록해주면 되겠습니다. <br><br>


```javascript

// Authentication/authentication.module.ts

import { Module } from '@nestjs/common';
import { AuthenticationService } from './authentication.service';
import { UsersModule } from '../users/users.module';
import { AuthenticationController } from './authentication.controller';
import { PassportModule } from '@nestjs/passport';
import { LocalStrategy } from './local.strategy';
 
@Module({
  imports: [UsersModule, PassportModule],
  providers: [AuthenticationService, LocalStrategy],
  controllers: [AuthenticationController]
})
export class AuthenticationModule {}


// authentication/authentication.controller.ts

import { Body, Req, Controller, HttpCode, Post, UseGuards } from '@nestjs/common';
import { AuthenticationService } from './authentication.service';
import RegisterDto from './dto/register.dto';
import RequestWithUser from './requestWithUser.interface';
import { LocalAuthenticationGuard } from './localAuthentication.guard';
 
@Controller('authentication')
export class AuthenticationController {
  constructor(
    private readonly authenticationService: AuthenticationService
  ) {}
 
  @Post('register')
  async register(@Body() registrationData: RegisterDto) {
    return this.authenticationService.register(registrationData);
  }
 
  @HttpCode(200)
  @UseGuards(LocalAuthenticationGuard)
  @Post('log-in')
  async logIn(@Req() request: RequestWithUser) {
    const user = request.user;
    user.password = undefined;
    return user;
  }
}
```

## 2. passport와 jwt를 이용하여 `Authorization` 
인증된 사용자는 매번 어플리케이션에 접속하고 뭔가 요청을 보낼때마다 로그인할 수 는 없습니다. 우리는 이런 귀찮은 것들을 막기 위해 `jwt` 토큰을 사용하여 사용자에게 어떤 권한이 있는지 검사하면 됩니다.
`jwt` 토큰을 사용할 때는 2개의 거의 필수적인 변수들이 필요한데, `JWT_SECRET` 값과 `JWT_EXPIRATION_TIME` 입니다.
`JWT_SECRET` 은 절대 노출되어서는 안됩니다. 이를 이용해 토큰을 decode하거나 encode하여 어플리케이션에 영향을 줄 수 있기 때문입니다. `JWT_EXPIRATION_TIME` 도 너무 길거나 너무 짧게 가져가면 안됩니다. 만료기한이 너무 길다면, 그 안에 유출될 가능성이 있고, 너무 짧다면 사용자가 여러번 로그인을 수행해야 합니다.

관련 패키지
<br>
```
npm install @nestjs/jwt passport-jwt @types/passport-jwt cookie-parser @types/cookie-parser
```

authentication module에서 jwt module을 사용해야 하므로, authentication module에 import 해줍니다.

```javascript
@Module({
  imports: [
    UsersModule,
    PassportModule,
    ConfigModule,
    JwtModule.registerAsync({
      imports: [ConfigModule],
      inject: [ConfigService],
      useFactory: async (configService: ConfigService) => ({
        secret: configService.get('JWT_SECRET'),
        signOptions: {
          expiresIn: `${configService.get('JWT_EXPIRATION_TIME')}s`,
        },
      }),
    }),
  ],
  providers: [AuthenticationService, LocalStrategy],
  controllers: [AuthenticationController]
})
```

authentication service 에서 요청받은 id를 jwt 토큰으로 바꿔주는 메서드를 생성해줍니다. 

**authentication/authentication.service.ts**
```javascript
public getCookieWithJwtToken(userId: number) {
    const payload: TokenPayload = { userId };
    const token = this.jwtService.sign(payload);
    return `Authentication=${token}; HttpOnly; Path=/; Max-Age=${this.configService.get('JWT_EXPIRATION_TIME')}`;
  }
```

추가로 passport에서 cookie로 부터 jwt를 읽어서 관련 권한이 있는지 validate 하도록 구성해야 합니다.

**authentication/jwt.strategy.ts**
```javascript
import { ExtractJwt, Strategy } from 'passport-jwt';
import { PassportStrategy } from '@nestjs/passport';
import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Request } from 'express';
import { UsersService } from '../users/users.service';
import TokenPayload from './tokenPayload.interface';
 
@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(
    private readonly configService: ConfigService,
    private readonly userService: UsersService,
  ) {
    super({
      jwtFromRequest: ExtractJwt.fromExtractors([(request: Request) => {
        return request?.cookies?.Authentication;
      }]),
      secretOrKey: configService.get('JWT_SECRET')
    });
  }
 
  async validate(payload: TokenPayload) {
    return this.userService.getById(payload.userId);
  }
}
```

위 코드는 JWT 토큰을 `request?.cookies?.Authentication` 으로 부터 읽어서 decode하고 나온 id에 해당하는 값이 데이터베이스에 있는지 확인합니다. 

이렇게 생성한 JwtStrategy를 Authentication service에서 이용할 수 있도록 module에 등록해줍니다.

추가로, AuthGuard('jwt')를 인증이 필요한 endpoint의 미들웨어로 등록해줍니다.

예를 들어, 게시글을 작성하는 endpoint가 있다면, 
```javascript

@Controller('posts')
export default class PostsController {
  constructor(
    private readonly postsService: PostsService
  ) {}
 
  @Post()
  @UseGuards(JwtAuthenticationGuard)
  async createPost(@Body() post: CreatePostDto) {
    return this.postsService.createPost(post);
  }
 
  // (...)
}
```

로그아웃을 하는 endpoint가 있다면, 

```javascript
@UseGuards(JwtAuthenticationGuard)
  @Post('log-out')
  async logOut(@Req() request: RequestWithUser) {
    response.setHeader('Set-Cookie', this.authenticationService.getCookieForLogOut());
    return response.sendStatus(200);
}
```

마지막으로 NestJS에서 Guard가 어떤 순서상에서 동작하는 지 보여드릴 그림을 가져왔습니다. 이 그림을 끝으로 passport를 이용한 Authentication / Authorization 글을 마칩니다. 감사합니다.<br><br>

{{< figure src="/images/nestjs_lifecycle.png" title="NestJS LifeCycle (figure)" >}}