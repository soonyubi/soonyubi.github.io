# Unit Test


## Unit Test (by ChatGPT)
유닛 테스트(Unit test)는 소프트웨어 개발에서 사용되는 테스트의 한 종류로, 개별적인 소스 코드의 최소 단위인 유닛(일반적으로 함수, 메서드, 클래스 등)이 의도한대로 동작하는지 검증하는 테스트입니다. 유닛 테스트는 소스 코드의 특정 부분을 격리시켜 테스트하므로, 다양한 입력값과 상황을 시뮬레이션하여 코드의 정확성, 예외 처리, 경계 조건 등을 확인할 수 있습니다.

유닛 테스트의 목적은 다음과 같습니다:

1. `코드의 정확성 검증`: 유닛 테스트를 통해 코드가 의도한대로 동작하는지 확인하여 버그와 오류를 발견하고 수정할 수 있습니다.
2. `코드의 안정성 향상`: 유닛 테스트를 통해 코드의 예외 처리, 경계 조건 등을 확인하여 코드의 안정성을 향상시킬 수 있습니다.
3. `리팩토링 지원`: 유닛 테스트는 코드를 리팩토링할 때 기존 기능이 올바르게 작동하는지 확인하는 데 도움을 줍니다.
개발 속도 향상: 유닛 테스트를 통해 버그를 조기에 발견하고 수정하므로, 생산성을 향상시킬 수 있습니다.
4. `협업과 유지보수 용이성 향상`: 유닛 테스트는 코드의 동작을 명확하게 문서화하고, 다른 개발자가 코드를 이해하고 유지보수하는 데 도움을 줍니다.

<br>
일반적으로 좋은 유닛 테스트는 독립적으로 실행 가능하며, 빠르게 실행되어야 하며, 정확하고 명확한 검증 로직을 가져야 합니다. 유닛 테스트는 개발자가 작성하며, 소프트웨어 개발의 품질과 안정성을 높이는 중요한 도구로 사용됩니다.

## Example

보통 대부분의 서비스에 존재하는, Authentication/ User service 가 내가 의도하는대로 동작하는지 확인하기 위한 unit test를 작성하려고 한다.

### 실제 객체를 선언하면서 유닛테스트 작성

```javascript
describe('The AuthenticationService', () => {
  let authenticationService : AuthenticationService;
  beforeEach(()=>{
    authenticationService = new AuthenticationService(
      new UsersService(
        new Repository<User>()
      ),
      new JwtService({
        secretOrPrivateKey:"key"
      }),
      new ConfigService(),
    );
  })
  
  describe('when creating a cookie', () => {
    it('should return a string', () => {
      const userId = 1;
      expect(
        typeof authenticationService.getCookieWithJwtToken(userId)
      ).toEqual('string')
    })
  })
});
```

위와 같이 코드를 구성할 경우, Authentication Service에 인스턴스를 생성할 때마다 필요한 dependency를 수동으로 넣어줘야 하므로, 올바른 작성방법은 아니다.

<br>

### dependency를 수동으로 넣어야 하는 이슈 해결 
그러므로, 우리는 `@nestjs/testing` 라이브러리의 `Test.createTestingModule().compile()` 메서드를 사용해 dependency를 수동으로 넣어줘야 하는 문제를 해결할 수 있다.

```javascript
describe('The AuthenticationService', () => {
  let authenticationService : AuthenticationService;
  beforeEach(async ()=>{
    const module = await Test.createTestingModule({
      imports: [
        UsersModule,
        ConfigModule.forRoot({
          validationSchema: Joi.object({
            POSTGRES_HOST: Joi.string().required(),
            POSTGRES_PORT: Joi.number().required(),
            POSTGRES_USER: Joi.string().required(),
            POSTGRES_PASSWORD: Joi.string().required(),
            POSTGRES_DB: Joi.string().required(),
            JWT_SECRET: Joi.string().required(),
            JWT_EXPIRATION_TIME: Joi.string().required(),
            PORT: Joi.number(),
          })
        }),
        DatabaseModule,
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
      providers: [
        AuthenticationService
      ],
    }).compile();
    authenticationService = await module.get<AuthenticationService>(AuthenticationService);
  })
  
  describe('when creating a cookie', () => {
    it('should return a string', () => {
      const userId = 1;
      expect(
        typeof authenticationService.getCookieWithJwtToken(userId)
      ).toEqual('string')
    })
  })
});
```
### 실제 DB를 사용하지 않고 DB mock 객체를 생성 

위와 같이 Test Module 필요한 module들을 미리 import해서 dependency를 매번 수동으로 넣어줘야 하는 문제를 해결할 수 는 있지만, 현재 import되는 모듈 중 Database Module은 실제 DB를 의미하기 때문에 테스트에 적합하지 않다. 
따라서 우리는 DB Module을 `mocking` 해서 provider로 제공해줘야 한다. 

```javascript
describe('The AuthenticationService', () => {
  let authenticationService : AuthenticationService;
  beforeEach(async ()=>{
    const module = await Test.createTestingModule({
      imports: [
        UsersModule,
        ConfigModule.forRoot({
          validationSchema: Joi.object({
            POSTGRES_HOST: Joi.string().required(),
            POSTGRES_PORT: Joi.number().required(),
            POSTGRES_USER: Joi.string().required(),
            POSTGRES_PASSWORD: Joi.string().required(),
            POSTGRES_DB: Joi.string().required(),
            JWT_SECRET: Joi.string().required(),
            JWT_EXPIRATION_TIME: Joi.string().required(),
            PORT: Joi.number(),
          })
        }),
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
      providers: [
        UsersService,
        AuthenticationService,
        {
          provide:getRepositoryToken(User),
          useValue:{}
        }
      ],
    }).compile();
    authenticationService = await module.get<AuthenticationService>(AuthenticationService);
  })
  
  describe('when creating a cookie', () => {
    it('should return a string', () => {
      const userId = 1;
      expect(
        typeof authenticationService.getCookieWithJwtToken(userId)
      ).toEqual('string')
    })
  })
});
```

위와 같이 mocked User Repository를 제공하기 위해 
```
{
    provide:getRepositoryToken(User),
    useValue:{}
}
```
해당 객체를 provider에 제공해줬다. 추가로, UserRepository는 User Module에서 `TypeormModule.forFeature([User])`로 import 되고 있기 때문에, UserService도 provider로 제공한 것을 볼 수 있다.
<br>

### ConfigService / JwtService에 대한 mock 객체 생성
여기서 더 나아가서, `ConfigModule` 과 `JwtModule을` 직접 import하는게 아닌, 각각을 mock 객체로 생성해서 이를 provider로 제공하는 작업을 처리하겠다.

```javascript
const mockedConfigService = {
  get(key: string) {
    switch (key) {
      case 'JWT_EXPIRATION_TIME':
        return '3600'
    }
  }
}

const mockedJwtService = {
  sign: () => ''
}

describe('The AuthenticationService', () => {
  let authenticationService : AuthenticationService;
  beforeEach(async ()=>{
    const module = await Test.createTestingModule({
      providers: [
        UsersService,
        AuthenticationService,
        {
          provide:getRepositoryToken(User),
          useValue:{}
        },
        {
          provide:ConfigService,
          useValue: mockedConfigService
        },
        {
          provide:JwtService,
          useValue:mockedJwtService
        }
      ],
    }).compile();
    authenticationService = await module.get<AuthenticationService>(AuthenticationService);
  })
  
  describe('when creating a cookie', () => {
    it('should return a string', () => {
      const userId = 1;
      expect(
        typeof authenticationService.getCookieWithJwtToken(userId)
      ).toEqual('string')
    })
  })
});
```

### Jest.fn() 을 이용해 mock 객체 생성
User Service도 `jest.fn()` 을 이용해 mock 객체를 생성해서 dependency를 직접 생성하지 않고 해결할 수 있습니다. 

```javascript
describe('The UsersService', () => {
  let usersService: UsersService;
  let findOne: jest.Mock;
  beforeEach(async () => {
    findOne = jest.fn();
    const module = await Test.createTestingModule({
      providers: [
        UsersService,
        {
          provide: getRepositoryToken(User),
          useValue: {
            findOne
          }
        }
      ],
    })
      .compile();
    usersService = await module.get(UsersService);
  })
  describe('when getting a user by email', () => {
    describe('and the user is matched', () => {
      let user: User;
      beforeEach(() => {
        user = new User();
        findOne.mockReturnValue(Promise.resolve(user));
      })
      it('should return the user', async () => {
        const fetchedUser = await usersService.getByEmail('test@test.com');
        expect(fetchedUser).toEqual(user);
      })
    })
    describe('and the user is not matched', () => {
      beforeEach(() => {
        findOne.mockReturnValue(undefined);
      })
      it('should throw an error', async () => {
        await expect(usersService.getByEmail('test@test.com')).rejects.toThrow();
      })
    })
  })
});
```
