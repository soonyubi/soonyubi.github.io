---
title: "[Project] Youtube Shorts Generator : dall3 api 를 쓰면서 "
subtitle: ""
date: 2024-04-16T01:05:39+09:00
lastmod: 2024-04-16T01:05:39+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: ["project"]

featuredImage: ""
featuredImagePreview: ""

hiddenFromHomePage: false
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

`Youtube Shorts Generator` 는 content 를 기반으로 gpt에게 스토리보드를 작성하게 하고, 스토리보드는 여러개의 씬을 가지고 있다. 각 씬의 이미지 프롬프트를 기반으로 dalle-3 모델을 사용하여 각 스토리에 대한 이미지 생성시키고, 생성된 이미지를 기반으로 stability AI 의 img2vid 모델을 이용하여 이미지를 짧은 비디오로 변환하여 모든 씬을 이어 붙인 후, 대본을 입히고, 음성을 입혀 요즘 유행하는 shorts를 자동으로 생성하게 해주는 프로그램이다.

오늘 포스팅할 주제는, 작성된 스토리보드를 기반으로 이미지를 생성하는 dalle3 모델을 어떻게 사용했는지, 그리고 해당 모델을 사용하면서 생겼던 문제와 그 문제를 어떻게 해결하고자 하는지에 대한 포스팅을 하려고 한다.

```
DALL-E 3를 개발자 관점에서 보면, 이 모델은 자연어 처리와 이미지 생성 기술을 통합한 인공지능 모델이야. DALL-E 3의 핵심은 텍스트를 입력받아 그 내용에 부합하는 고해상도 이미지를 생성하는 건데, 이 과정에서 몇 가지 중요한 기술적 요소들이 작용해.

1. 모델 아키텍처
DALL-E 3는 변형된 트랜스포머 아키텍처를 사용해. 트랜스포머는 주로 자연어 처리에서 사용되지만, 이미지와 같은 다른 형태의 데이터에도 적용 가능하다는 점이 밝혀졌어. DALL-E 3는 이 아키텍처를 활용해 텍스트 입력을 이미지 데이터로 변환하는 복잡한 과정을 처리해.

2. 훈련 데이터와 학습 방식
이 모델은 대규모 데이터셋에서 학습되어야 고품질의 이미지를 생성할 수 있어. 훈련 데이터는 다양한 텍스트 설명과 해당 설명에 맞는 이미지로 구성되어, 모델이 텍스트와 이미지 사이의 관계를 학습할 수 있게 해줘. 이 과정에서는 높은 컴퓨팅 리소스와 복잡한 데이터 전처리 작업이 필요해.

3. Zero-shot, Few-shot Learning
DALL-E 3는 zero-shot 또는 few-shot 학습 능력을 갖추고 있어서, 본 적 없는 새로운 텍스트 설명에 대해서도 적절한 이미지를 생성할 수 있어. 이는 모델이 광범위하게 일반화하는 능력을 가지고 있음을 의미해.

4. 개선된 해상도와 디테일
이전 모델들에 비해 DALL-E 3는 이미지의 해상도와 디테일이 크게 향상되었어. 이는 복잡한 텍스처, 섬세한 디테일, 정확한 색상 재현 등을 가능하게 해서, 사용자가 원하는 더 사실적이고 세밀한 이미지를 생성할 수 있게 해줘.

5. 응용 프로그램 및 API 제공
OpenAI는 DALL-E 3를 활용할 수 있게 API를 제공하고 있어. 개발자는 이 API를 통해 자신의 애플리케이션, 웹사이트, 소프트웨어 등에 DALL-E 3 기능을 쉽게 통합하고, 다양한 용도로 활용할 수 있어.

by gpt.
```

나는 openAI 에서 dall3 모델을 사용할 수 있는 API를 사용했다. API를 호출하는 방법은 단순히 API_KEY 를 어플리케이션에 등록만 해주면 되기 때문에 사용이 간편하다.

```node.js
from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="a white siamese cat",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
```

`dall-e-3` 모델을 사용하게 되면, quality 값은 기본적으로 'standard' 인데, 'hd'의 고해상도 이미지를 생성하도록 할 수 있다.
추가적으로 style을 지정할 수 있는데, 'vivid' 와 'natural'을 제공한다. vivid 는 hyper-real, dramastic image 를 생성한다고 하고, natural은 말 그대로 좀 더 자연스럽게 이미지를 생성한다.

사이즈를 자유롭게 선택할 수 있다면 좋겠지만, 아쉽게도 dalle3는 1024x1024, 1024x1792 or 1792x1024 pixels 만 지원한다. 따라서, 해당 프로젝트의 이미지 생성 모델은 dalle3가 생성한 이미지를, img2vid 모델의 input에 맞게 해상도를 변경해야 하므로 해상도를 변경하는 기능을 추가해줬다.

dall3 는 웹에서 접근할 수 있는 이미지 url 또는 base64 json 포맷으로 이미지를 응답하는데, 웹 url은 접근 제한 시간이 1시간임에 유의하자.

여기서 n에 주목을 해보면, n값은 모델이 생성하는 이미지의 갯수이다. dalle3의 모델의 경우 n=1만을 지원한다.dalle2 모델을 사용할 경우, 프롬프트당 10개의 이미지가 생성이 가능하다.

여기서 또 하나 문제가 있는데, openAI 에 등록된 API KEY 당 1분당 dalle3 API 호출횟수에 제한이 있다는 것이었다.
openAI 에는 tier 가 존재해서, 각 티어마다 분당 api 호출횟수의 제한이 있다. 나는 돈이 없는 백수이기 때문에, 1달에 50달러 이상을 openAI 투자할 수 없어 1분당 호출횟수는 5번이 제한이다. (dall3 모델의 경우)

gpt 모델을 사용하여 생성한 스토리보드는 10개의 씬으로 구성되어 있고, 각 씬 별로 이미지를 생성하도록 호출했더니 rate limit 제한에 걸려 이미지를 생성하지 못하는 이슈가 있었다.

물론 해당 프로그램을 나 혼자서 쓰게 된다면 문제는 없지만, 나는 이 프로그램을 다른 사람들이 쓸 수 있도록 만들고 싶었고, 1분당 호출횟수 제한의 벽을 뚫기 위해 어떻게 하면 좋을 지 생각해보았다.
첫번째로 든 생각은, 각 씬에 대한 이미지 프롬프트를 받는 대기열 큐에 이미지 프롬프트를 등록하고, 1분마다 cron job 을 돌려서 5개씩 메시지를 빼오면 좋겠다는 생각을 했다.
여기서 또 하나 들었던 생각은 5개의 메시지가 모두 한 사람이 생성한 프롬프트가 아닐 수 있기 때문에 모든 씬에 대한 이미지 생성 후 작동할 img2vid 모듈이 언제 비디오를 생성하면 좋을 지 모른다는 것이었다.

그래서 두번째로 했던 생각은, 최근에 카프카 관련 책을 읽으면서 control message 를 이용해 트랜잭션 관리를 하는 내용을 본적이 있어서, control message 와 비슷하게 각 스토리 보드 마지막 씬의 이미지 프롬프트를 메시지큐에 넣고 나서, 해당 이미지 프롬프트가 마지막이라는 것을 알릴 수 있도록 메시지 큐에 "특정 스토리보드의 이미지 프롬프트의 끝입니다" 라는 메시지를 하나 더 추가해주면 위 문제가 해결될 수도 있겠다라는 생각을 했다.

추가적으로, 이미지 생성 모델이 dalle3 만 존재하는 것이 아니고 여러 모델이 존재하기 때문에 ( e.g. stable diffusion) 위 모델을 같이 사용하게 되면, 분당 5개 이상의 이미지를 생성할 수 있겠다는 생각을 했다.
더 실험해봐야 할 것은, 분당 호출횟수를 늘리기 위해, 서로 다른 ip, api-key를 가진 서버를 여러개 띄워서 하는 방법도 시도해보려고 한다.
