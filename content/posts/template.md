---
title: "Nestjs: Gaurd를 이용해 Authentication/ Authorization 구현"
author: Richard
date: 2023-04-05T20:50:23+09:00
categories:
    # - backend
tags:
    # - nest.js
    # - guard
    # - authentication
    # - authorization
featuredImage: 
images :
draft: false
---

## big picture
[v] passport와 bcrypt를 이용하여 사용자 인증
    


## 1. passport와 bcrypt를 이용하여 사용자 인증

User module에 User Entity를 생성하고, User service에서 User database에 create / fetch 하는 메서드를 간단히 만들어 줍니다. UserService를 Authentication module에서 사용하길 원하기 때문에, UserService를 `@Injectable()` 데코레이터로 감싸주고, UserModule에서 UserService를 export해줍니다. 

password는 가장 안전해야 하는 데이터입니다. 그래서 password는 `hash` 해야 합니다. `hash`를 하는 과정에서 필요한 값은 random string 인 `salt` 값이 필요합니다. 

이 모든 과정을 bcrypt 라이브러리를 사용하면 쉽게 할 수 있습니다. bcrypt로 password에 salt값을 적용해 여러번 hash하여 복원하는 것을 어렵게 합니다. bcrypt는 cpu를 잡아먹는 작업이지만, thread pool의 추가적인 thread를 이용해 연산을 수행하므로 암호화하는 과정에서 다른 작업을 수행할 수 있습니다.

<!-- annotation note -->
{{< admonition note "some title" >}}

{{< /admonition >}}

<!-- annotation tip -->
{{< admonition tip "some title" >}}

{{< /admonition >}}

{{< admonition question "some title" >}}

{{< /admonition >}}

<!-- note abstract info tip success question warning failure danger bug example anotation 이 있음 -->

<!-- # img -->
![Basic configuration preview](basic-configuration-preview.png "Basic configuration preview")

<!-- emphasize -->
**hi**

<!-- Fraction -->
[Light]/[dark]

[link{?]}(hello)

<!-- Youtube -->
{{< youtube w7Ft2ymGmfc >}}

<!-- Figure -->
{{< figure src="/images/lighthouse.jpg" title="Lighthouse (figure)" >}}

<!-- 특정 사람을 재밌게 넣고 싶을 때 -->
{{< person url="https://evgenykuznetsov.org" name="Evgeny Kuznetsov" nick="nekr0z" text="author of this shortcode" picture="https://evgenykuznetsov.org/img/avatar.jpg" >}}

<!-- Task List -->
- [x] Write the press release
- [ ] Update the website
- [ ] Contact the media

<!-- Tables -->
<!-- :을 이용해서 정렬 -->
| Option | Description |
|:------:| -----------:|
| data   | path to data files to supply the data that will be passed into templates. |
| engine | engine to be used for processing templates. Handlebars is the default. |
| ext    | extension to be used for dest files. |

<!-- Link -->
[Assemble](https://assemble.io)