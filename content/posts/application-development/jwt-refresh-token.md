---
title: "Nest.js : Jwt Refresh Token"
subtitle: ""
date: 2023-04-20T01:01:23+09:00
lastmod: 2023-04-20T01:01:23+09:00
draft: false
author: "홍순엽"
authorLink: ""
description: ""
license: ""
images:
  [
    "/images/JWT-Refresh-Token-workflow-diagram-aspnet-core_637779427789390135.png",
  ]

tags: ["nest.js", "jwt", "refresh-token"]
categories: ["application-development"]

featuredImage: ""
featuredImagePreview: ""

hiddenFromHomePage: true
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

유저가 로그인 했을 때, `access token` 만을 이용해서 권한을 인증하고, 만료가 되면 다시 로그인하는 과정을 거친다면 무슨 일이 일어날까? <br>
만약 만료시간이 짧다면, 유저는 재로그인해야 하는 상황이 계속 올 수 밖에 없다. 그러면 유저입장에서 불편할 수 있다. <br>
만약 access token이 공격자에게 탈취된다면, 공격자는 로그인 없이도 모든 서비스에 접근 권한을 얻을 수 있다. 이것은 상당한 문제이다. <br><br>

이를 해결하기 위해선 `refresh-token` 을 하나더 생성하면 된다. `refresh-token` 은 `access-token` 을 재발급할 수 있는 토큰이다. <br>

그럼, `refresh-token` 이 탈취된다면 어떻게 해야할까? <br>
`refresh-token` 은 `stateless` 이기 때문에 가장 간단하게 생각해볼 수 있는 방법으로는, secret 값을 변경해서 모든 토큰을 무효화 할 수 있다. 하지만 이 방법은 모든 유저에게 다시 로그인하게 하는 대참사가 일어날 수 있기 때문에 현명한 방법은 아니다. <br>

하나의 해결책은 `refresh-token` 을 db에 저장해서 관리하고 로그인할 때마다 변경하면 된다. 이런 방법을 채택하면, 다음과 같은 문제도 해결할 수 있다. <br>
예를 들어, 여러 사용자가 하나의 계정을 공유한다고 했을 때, 모든 사람이 갑자기 해당 계정으로 동시에 접속해 어플리케이션을 이용한다면, 비지니스에 안좋은 영향을 끼칠 수 있다. `refresh-token` 을 로그인할 때마다 변경한다면, 이전에 로그인한 사용자는 유효하지 않은 `refresh-token`을 갖게 될 것이고, 재로그인을 유도할 것이다.
<br>

그럼 `refresh-token`이 db에서 유출되는 문제는 어떻게 해결해야 할까? <br>
password 처럼 `hash` 해서 저장하면 된다.

{{< image src="/images/JWT-Refresh-Token-workflow-diagram-aspnet-core_637779427789390135.jpg" caption="" width="100%" >}}

## 구현

### configuration

```
// .env
JWT_ACCESS_TOKEN_SECRET=access-secret
JWT_ACCESS_TOKEN_EXPIRATION_TIME=3600
JWT_REFRESH_TOKEN_SECRET=refresh-secret
JWT_REFRESH_TOKEN_EXPIRATION_TIME=3600

// app.module.ts
      ....
    JWT_ACCESS_TOKEN_SECRET: Joi.string().required(),
    JWT_ACCESS_TOKEN_EXPIRATION_TIME: Joi.string().required(),
    JWT_REFRESH_TOKEN_SECRET: Joi.string().required(),
    JWT_REFRESH_TOKEN_EXPIRATION_TIME: Joi.string().required(),
    ....
```

위와 같이 `access token` / `refresh token` 에 대해 변수를 생성해주고, 필요하다면 이를 validation까지 하는 로직을 구현해주면 된다.

### /login

##### access token 발급

```javascript
async getJwtAccessToken(userId: number){
  const payload : TokenPayload = {userId};
  const token = await jwtService.sign(payload, {
    secret : configService.get("JWT_ACCESS_TOKEN_SECRET"),
    expiresIn : configService.get("JWT_ACCESS_TOKEN_EXPIRATION_TIME")
  });
}
```

##### refresh token 발급

```javascript
async getJwtRefreshToken(userId: number){
  const payload : TokenPayload = {userId};
  const token = await jwtService.sign(payload, {
    secret : configService.get("JWT_REFRESH_TOKEN_SECRET"),
    expiresIn : configService.get("JWT_REFRESH_TOKEN_EXPIRATION_TIME")
  });
}
```

{{< admonition note "signOptions" >}}
현재 위 두 로직에서 sign() 메서드를 실행할 때 `signOption`에 `secret` 값이 들어가있는 걸 볼 수 있다. 이렇게 함으로써 다른 secret값에 따라 다르게 토큰을 생성하는데 관련 기능은 `@nest/jwt@^7.1.0` 에서부터 가능하다.
{{< /admonition >}}

##### user db에 현재 토큰을 hash 하고 저장

다음과 같이 현재 들어온 refreshToken을 새로 암호화하고 저장하므로, refresh token이 탈취되었을 때 발생하는 문제를 어느정도 해결할 수 있다.

```javascript
// userService.ts
async setCurrentRefreshToken(refreshToken: string, userId: number) {
    const currentHashedRefreshToken = await bcrypt.hash(refreshToken, 10);
    await this.usersRepository.update(userId, {
      currentHashedRefreshToken
    });
  }
```

##### cookie에 access token, refresh token을 넣고 response

```javascript
@HttpCode(200)
  @UseGuards(LocalAuthenticationGuard)
  @Post('log-in')
  async logIn(@Req() request: RequestWithUser) {
    const {user} = request;
    const accessTokenCookie = this.authenticationService.getCookieWithJwtAccessToken(user.id);
    const refreshTokenCookie = this.authenticationService.getCookieWithJwtRefreshToken(user.id);

    await this.usersService.setCurrentRefreshToken(refreshToken, user.id);

    request.res.setHeader('Set-Cookie', [accessTokenCookie, refreshTokenCookie]);
    return user;
  }
```

### /refresh

##### refresh token validation

현재 refresh token이 validate 하는지 확인하기 위해서, 새로운 strategy를 생성해야 한다. 이 strategy를 `JwtRefreshStrategy` 라 한다면, 이 strategy에서는 user service에 현재 들어온 token이 유저 db에 저장된 토큰과 똑같은 지 비교한다. 이 과정을 통해서, token이 갈취되거나 동일한 사용자가 동시에 접속하는 문제를 해결할 수 있다.

```javascript
//
@Injectable()
export class JwtRefreshTokenStrategy extends PassportStrategy(
  Strategy,
  'jwt-refresh-token'
) {
  constructor(
    private readonly configService: ConfigService,
    private readonly userService: UsersService,
  ) {
    super({
      jwtFromRequest: ExtractJwt.fromExtractors([(request: Request) => {
        return request?.cookies?.Refresh;
      }]),
      secretOrKey: configService.get('JWT_REFRESH_TOKEN_SECRET'),
      passReqToCallback: true,
    });
  }

  async validate(request: Request, payload: TokenPayload) {
    const refreshToken = request.cookies?.Refresh;
    return this.userService.getUserIfRefreshTokenMatches(refreshToken, payload.userId);
  }
}
```

`passReqToCallback` 을 통해 `validate` 메서드에서 cookie에 접근할 수 있게 되었다.

```javascript
// user service
async getUserIfRefreshTokenMatches(refreshToken: string, userId: number) {
    const user = await this.getById(userId);

    const isRefreshTokenMatching = await bcrypt.compare(
      refreshToken,
      user.currentHashedRefreshToken
    );

    if (isRefreshTokenMatching) {
      return user;
    }
  }
```

```javascript
@UseGuards(AuthGuard('jwt-refresh-token'))
  @Get('refresh')
  refresh(@Req() request: RequestWithUser) {
    const accessTokenCookie = this.authenticationService.getCookieWithJwtAccessToken(request.user.id);

    request.res.setHeader('Set-Cookie', accessTokenCookie);
    return request.user;
  }
```

### /logout

로그아웃은 유저db에 저장된 refresh token을 없애고, response로 token을 무효화하는 정보를 저장해 응답한다.

```javascript
// authentication service
public getCookiesForLogOut() {
    return [
      'Authentication=; HttpOnly; Path=/; Max-Age=0',
      'Refresh=; HttpOnly; Path=/; Max-Age=0'
    ];
  }

// user service
async removeRefreshToken(userId: number) {
    return this.usersRepository.update(userId, {
      currentHashedRefreshToken: null
    });
  }

// authentication controller
@UseGuards(JwtAuthenticationGuard)
  @Post('log-out')
  @HttpCode(200)
  async logOut(@Req() request: RequestWithUser) {
    await this.usersService.removeRefreshToken(request.user.id);
    request.res.setHeader('Set-Cookie', this.authenticationService.getCookiesForLogOut());
  }
```
