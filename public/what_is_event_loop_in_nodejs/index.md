# Node.js 이벤트 루프


## 개요

면접에서 이벤트 루프가 뭐냐는 질문에 전혀 대답하지 못해서 찾아보다가 여러 커뮤니티에서 이벤트 루프의 정확한 개념에 대해 심도 있게 생각하는 걸 보고 정리하고자 하였습니다. 

## setTimeout vs setImmediate

```javascript
setTimeout(() => {
    console.log("setTimeout")
}, 0)
setImmediate(() => {
    console.log("setImmediate")
})
```

위 코드를 실행하면 뭐가 먼저 결과로 출력될까? 정답은 모른다 입니다. 그 이유를 설명하기 위해선 이벤트 루프의 구조를 살펴봐야 합니다.

## Node.js 이벤트 루프

Node.js를 `싱글 스레드 논 블로킹` 이라고 합니다. 단일 스레드인데 I/O 작업이 발생한 경우 이를 비동기로 처리할 수 있다는 의미입니다. 싱글 스레드면 하나의 작업이 끝날 때까지 기다려야 하는데 왜 빠른가에 대한 이유는 이벤트 루프가 존재하기 때문입니다. 

## Node.js 구조

{{< figure src="/images/nodejs_structure.jpg" title="Node.JS Structure (figure)" >}}

Node에서는 비동기 처리를 하기 위해 `이벤트 루프` 기법을 사용합니다. 이는 `libuv` 라는 라이브러리 내에 c언어로 구현되어 있습니다. 
<br>
```c++

// deps/uv/src/unix/core.c

 while (r != 0 && loop->stop_flag == 0) {
      uv__update_time(loop); // loop time 갱신
      uv__run_timers(loop); // timers 이벤트 처리
      ran_pending = uv__run_pending(loop); // IO callbacks 이벤트큐 처리
      uv__run_idle(loop);
      uv__run_prepare(loop);

      timeout = 0;
      if ((mode == UV_RUN_ONCE && !ran_pending) || mode == UV_RUN_DEFAULT)
          timeout = uv_backend_timeout(loop);

      uv__io_poll(loop, timeout); // poll 이벤트큐 처리
      uv__run_check(loop); // check 이벤트큐 처리
      uv__run_closing_handles(loop); // close 이벤트큐 처리

      r = uv__loop_alive(loop); // 이벤트루프 상태체크
      if (mode == UV_RUN_ONCE || mode == UV_RUN_NOWAIT)
          break;
  }
```

<br>

`libuv`에게 비동기 작업을 요청하게 되면 `libuv`는 이 작업이 커널에서 지원하는지 확인하고 지원한다면 커널에 해당 작업을 요청하고 응답을 받습니다. 지원하지 않는다면, 워커 스레드를 이용해서 작업을 처리하게 됩니다. 

<br>

다시 처음으로 돌아가서, `Single thread Non-blocking I/O`를 정의하자면, Node.js 는 I/O 작업을 메인 스레드가 아닌 워커 스레드에 위임함과 동시에 `Event-Loop` 라는 기법을 통해 `Non-blocking I/O`를 지원합니다. 

<br>

`Event-Loop`는 다음과 같은 Phase를 거칩니다. 그리고 각 Phase에는 자신의 큐를 가지고 있습니다.
- `Timer` : setTimeout(), setInterval() 에 의해 스케쥴된 callback들이 수행
- `Pending` : 이전 이벤트에서 수행되지 못한 I/O callback을 처리
- `idle/prepare` : 내부적으로 사용 (tick frequency 관리 ?)
- `Poll` : close / timer / setImmediate() 를 제외한 거의 모든 콜백을 집행 (http, apiCall, db)
- `Check` : setImmediate()
- `Close` : socket.on('close',...) 같은 close callback들

{{< admonition question "" >}}
여러 블로그를 참고했는데 어느 블로그에선 FIFO 큐라고 하고, 어느 블로그에서는 min-Heap이라 해서 잘 모르겠지만, 대충 뭔갈 담아 놓는 공간이 있다는 것, 그리고 그것들을 특정한 기준으로 뽑아서 처리한다는 것 세부적인 것은 찾아봐야 할 듯 

[여기](https://tk-one.github.io/2019/02/07/nodejs-event-loop/) 를 참고했을 때 큐를 돌면서 실행하지 않고, 스택을 처리한다... 뭔소리지 ㅋㅋ
{{< /admonition >}}

## nextTickQueue / microTaskQueue

`process.nextTick()`은 `nextTickQueue` 가 관리하고, Promise의 Resolve 결과물은 `microTaskQueue` 가 관리합니다. 얘네들은 **지금 수행하고 있는 작업이 끝나면 그 즉시 수행**합니다.

그리고, `nextTickQueue의` 우선순위가 `microTaskQueue의` 우선순위보다 높습니다.

```javascript
setTimeout(() => {
    console.log(1)
    process.nextTick(() => {
        console.log(3)
    })
    Promise.resolve().then(() => console.log(4))
}, 0)
setTimeout(() => {
    console.log(2)
}, 0)
```

위 코드의 수행결과는 어떻게 될까? 노드 v11.0 이상 기준으로 1-3-2-4가 된다.

Timer Phase에 먼저 console.log(1)을 등록하고, Timer Phase에서 이 callback을 처리하는 순간 `nextTickQueue`, `microTaskQueue` 에 console.log(3)과 console.log(4)를 등록하고, 현재 수행할 작업이 없기 때문에 바로 처리가 됩니다.

## 마무리

`Event-Loop`에 대한 개념을 이렇게 심도있게 다뤄야하나 싶었지만, 이를 통해서 얻을 수 있었던 것은 만약 특정 API에 부하가 생기고 콜백 큐가 바빠져서 Event-Loop 가 고갈이 되었을 때 해결방법에 대해 찾아볼 수 있었습니다.

1. 스레드풀 조정
`UV_THREADPOOL_SIZE` 변수 값을 수정해서 스레드를 기본 4개에서 128개까지 조정할 수 있습니다. 이 방법은 I/O 작업이 많은 경우에 도움이 될 수 있겠지만, 큰 스레드풀은 CPU나 메모리를 고갈시킬 수 있음을 기억해야 합니다.

2. 만약 Node로 작성한 어플리케이션이 CPU를 많이 잡아먹는 환경에서 사용된다면, 이 특정 작업에 더 잘 맞는 언어를 선택해 작업량을 줄이는 방법도 생각해볼 수 있겠습니다.


