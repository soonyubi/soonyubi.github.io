# Schedule Email With Nodemailer


## Gmail 앱 비밀번호 설정
nodemailer를 사용하기 위해선, 이메일을 보내는 주체가 필요한데, 이를 위해서 이번 포스트에서 Gmail을 사용할 예정이다. nodemailer가 Gmail에 접근할 때 인증된 사용자인지 확인하기 위해 credentail 값에 구글 아이디 / 비밀번호를 제공하는데, 이때 비밀번호는 구글 계정에서 생성한 무작위의 영문자 16자로 이루어진 앱 비밀번호이다.
<br>
이를 생성하기 위해선 다음과 같은 조건이 만족되어야 한다.
- 회사, 학교, 조직의 계정이면 안된다.
- 2차 인증을 해야 사용할 수 있다.
- 이러고 나서도 화면에 표시되지 않았다면, 활성화된 `2단계 인증` 탭에 들어가 제일 하단에 앱 비밀번호를 생성해주면 된다.


## Nest에서 Task Scheduling
Linux에선 스케쥴링 작업을 실행하기 위해, `cron`을 사용한다. 이러한 기능은 node에도 존재한다. nest에서는 node의 `cron` 라이브러리 하에서 스케쥴링 작업을 실행할 수 있다.

```
$ npm install --save @nestjs/schedule
$ npm install --save-dev @types/cron
```

### register module 
```javascript
import { Module } from '@nestjs/common';
import { ScheduleModule } from '@nestjs/schedule';

@Module({
  imports: [
    ScheduleModule.forRoot()
  ],
})
export class AppModule {}
```

위와 같이 모듈을 등록하면, 앱 내에 존재하는 모든 스케쥴링 작업을 등록한다. 이러한 등록은 nest lifecycle 중 `onApplicationBootstrap `이 hook을 일으켜 실행된다.

### schedule decorator 선언하기

#### cron job

다음과 같이 `cron-pattern`, `CronExpression enum`, `Date object`를 이용해서 cron job을 생성할 수 있다.

- `@cron(10 * * * * *)` 
- `@cron(CronExpression.EVERY_30_SECONDS)`
- `@cron(new Date(Date.now()+10*1000))` : 10초 뒤에 한번 실행

#### cron-pattern

{{< figure src="/images/cron-pattern.png" title="cron-pattern" >}}

#### Interval
`@Interval(10000)` 은 앱이 실행되고 10초 마다 실행하도록 스케쥴링을 하는 데코레이터이다. `@Interval()` 데코레이터는 `setInterval()`을 기반하여 만들어졌다.

#### Timeouts
`@Timeout(10000)`은 앱이 실행되고 10 뒤에 한번 실행하도록 스케쥴링 하는 데코레이터이다. `@Timeout()` 데코레이터는 `setTimeout()`을 기반하여 만들어졌다.

### 동적으로 scheduler 사용하기

예를 들어, 회원가입 로직 중 인증된 사용자인지 확인하기 위해 이메일로 인증화면을 보내는 기능을 구현한다고 가정할 때, 이를 구현하기 위해선 인증된 사용자인지 확인하는 화면에 도달했을 때, `/email-scheduling/verify-user-with-email` api 에 해당 사용자에게 이메일을 보내도록 동적으로 작업을 수행할 수 있다.

이를 달성하기 위해선, 다음과 같이 `SchedulerRegistry` instance를 생성해주면 된다.

```javascript
constructor(private schedulerRegistry: SchedulerRegistry) {}

addCronJob(name: string, seconds: string) {
  const job = new CronJob(`${seconds} * * * * *`, () => {
    this.logger.warn(`time (${seconds}) for job ${name} to run!`);
  });

  this.schedulerRegistry.addCronJob(name, job);
  job.start();

  this.logger.warn(
    `job ${name} added for each minute at ${seconds} seconds!`,
  );
}
```

현재 위에서 cron job을 생성할 때, 첫번재 인자로 `cron-pattern`을 넘겨주었지만, `date object`나 `Moment object`를 넘겨줄 수 도 있다.

## Nodemailer 사용해서 정해진 시간에 이메일 보내는 기능 구현

### api route handler 
```javascript
@Controller('email-scheduling')
export class EmailSchedulingController {
  constructor(private readonly emailSchedulingService: EmailSchedulingService) {}
 
  @Post('schedule')
  @UseGuards(JwtAuthenticationGuard)
  async scheduleMail(@Body() emailSchedule : EmailScheduleDto){
    this.emailSchedulingService.scheduleEmail(emailSchedule); 
  }
}
```
### cron job creation
```javascript
scheduleEmail(emailSchedule : EmailScheduleDto){
        const date = new Date(emailSchedule.date);
        console.log(date);
        const job = new CronJob(date, ()=>{
            this.emailService.sendMail({
                to : emailSchedule.recipient,
                subject : emailSchedule.subject,
                text : emailSchedule.content
            });
        });
        console.log(job);
        this.schedulerRegistry.addCronJob(`${Date.now()}-${emailSchedule.subject}`,job);
        job.start();
    }
```
### send email by using nodemailer instance
```javascript
constructor(
        private readonly configService : ConfigService
    ){
        this.nodemailerTransport = createTransport({
            service : configService.get("EMAIL_SERVICE"),
            auth : {
                user : configService.get("EMAIL_USER"),
                pass: configService.get("EMAIL_PASSWORD")
            }
        });
    }

    sendMail(options : Mail.Options){
        return this.nodemailerTransport.sendMail(options);
    }
```

## 마무리 
전체 코드는 [다음](https://github.com/soonyubi/nestjs-typescript/tree/master/src)에서 확인할 수 있다. 
