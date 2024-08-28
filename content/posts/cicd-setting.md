---
title: "사내 CI/CD 도입기"
subtitle: ""
date: 2024-08-08T10:07:03+09:00
lastmod: 2024-08-08T10:07:03+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: []

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

<p align='center'>
<img src="/images/experience/seereal/old.png" width="80%"/>
</p>

## 기존의 아키텍쳐와 배포 과정

팀에 합류하여 코드베이스와 인프라 구성을 보았고, 배포는 어떻게 진행하는 지에 들었을 때 어떻게 수정해야 될지 감이 오지 않았다. 기존엔 위와 같이 API Gateway가 EC2의 public ip 로 요청을 나눠주는 형태로 구성되어 있었다. 개발 서버 / 운영 서버용 API Gateway가 나눠져 있었고, 각각의 ec2 서버에 접속해서 특정 포트에서 실행중인 프로세스를 죽이고, 새로 웹서버를 띄우는 방식으로 배포를 진행하고 있었다. 추가로 개발서버와 운영서버의 레포지토리가 나눠져있었다. 이러한 이슈들을 어떤 생각으로 해결했는지를 작성하고자 한다.

위와 같이 배포를 진행하게 되면, 다음과 같은 문제가 발생한다.

- 배포를 수동으로 진행하므로, 휴먼에러가 발생할 가능성이 높다.
- 배포하는 과정이 너무 길고 반복적인 작업이다.
- 서버를 배포하는 과정에서 짧은 시간이지만 다운타임이 발생한다.

## 레포 합치기

첫번째로 진행한 것은 레포 합치기이다. 개발서버와 운영서버의 레포가 나뉘게 되면, 변경 히스토리를 정확하게 판단할 수 없기 때문에 이 부분을 합치는게 가장 우선이라고 판단했다. 코드베이스를 diff를 해가며 확인해보니, 메인 레포의 경우 기능적으로 조금 더 앞서있었고, 개발 레포의 경우 특정 ODM 을 사용하기 위해 적용하는 단계였다. 그래서 나는 메인 레포를 기준으로 개발 레포를 버리기로 결정했다.

## Config 값 종속성 분리

각 서버에는 config 파일이 저장되어 있었고, 웹서버가 실행할 때 해당 값을 참조하는 방식으로 이루어졌다. 하지만 우리팀의 경우 서버의 갯수가 많았고 비슷한 config 값들을 공통으로 쓰고 있었기 때문에, 하나의 config 값이 바뀌거나 추가되는 경우 모든 서버에서 이를 반영해줘야 하는 문제가 있었다.
이렇게 관리포인트가 늘어나는 부분을 한곳으로 집중하기 위해, 나는 secret manager를 이용해 config 값을 관리하고, ci/cd 를 이용해 어플리케이션 배포 시 aws secret manager로 부터 환경변수를 주입받아 사용하기로 결정했다.

## downtime 이슈

ci/cd를 이용해 배포를 자동화하기 위해선, 기존에 돌아가는 서버의 프로세스를 죽여야만 가능했다. 물론 port를 바꿔서 해결할 수 도 있었겠지만, security group 수정, api gateway의 endpoint 수정 등 여러 부분을 수정해줘야 할 거 같아, 기존 port를 사용하되 다운타임이 발생하지 않도록 어떻게 구성하면 좋을까 생각했다.
기존의 api gateway가 ec2 의 public ip를 endpoint 로 쓰던 부분을 load balancer의 dns로 바꾸고, load balancer는 기존의 서버와 ci/cd를 적용할 새로운 서버에 트래픽을 나눠주게 되면 다운타임을 최소로 할 수 있지 않을까 생각했다.

물론 api gateway의 endpoint를 바꾸는 과정에서 다운타임이 발생하지 않을까 생각하지 않은 것은 아니었다. 그러나 다행이도, api gateway의 배포 속도는 빨랐고 내부적으로 이전의 요청은 이전의 endpoint 로 라우팅해주는 기능이 있어보였다.

그래서 아래와 같이 모든 서버에 대해 로드밸런서를 생성하고 도메인을 바꿔주는 작업을 통해 다운타임없이 성공적으로 배포를 마칠 수 있었다.

<p align='center'>
<img src="/images/experience/seereal/old.png" width="80%"/>
</p>

## 만약 ec2 서버가 죽는다면 ?

EC2서버가 죽어도 서비스가 가능하도록 하기 위해서 EC2서버가 죽었을 때 ECR로 부터 가장 최신의 이미지를 가져와 실행하도록 `user data`를 구성했다.
추가로, ec2 서버를 ASG로 생성하여 특정 인스턴스 갯수를 항상 유지할 수 있도록 구성했다.
아래는 ec2 서버가 죽고 나서 실행되는 user data script 이다.

```script
#!/bin/bash
# 업데이트 및 필수 패키지 설치
sudo apt-get update -y
sudo apt-get install -y awscli jq docker.io
sudo chmod 666 /var/run/docker.sock

# AWS CLI 구성
aws configure set default.region ap-northeast-2

# ECR 로그인
aws ecr get-login-password --region ap-northeast-2 | sudo docker login --username AWS --password-stdin ***.dkr.ecr.ap-northeast-2.amazonaws.com

# 비밀 값 가져오기
SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id [secret-id] --query SecretString --output text)

# 비밀 값들을 환경 변수로 추출


REPO_NAME=[server-name]
LATEST_TAG=$(aws ecr describe-images --repository-name $REPO_NAME --region ap-northeast-2 | jq -r '.imageDetails[] | select(.imageTags != null) | select(.imageTags | map(endswith("-main")) | any) | {imagePushedAt: .imagePushedAt, imageTags: .imageTags[]}' | jq -s 'sort_by(.imagePushedAt) | reverse | .[0].imageTags' | tr -d '"')


if [ -z "$LATEST_TAG" ]; then
  echo "Error: No development tag found for repository $REPO_NAME"
  exit 1
fi

# 기존 Docker 컨테이너 중지 및 제거
if sudo docker inspect [server-name] &> /dev/null; then
  sudo docker stop [server-name]
  sudo docker rm [server-name]
fi

# 최신 Docker 이미지 풀
docker pull ***.dkr.ecr.ap-northeast-2.amazonaws.com/$REPO_NAME:$LATEST_TAG

# Docker 컨테이너 실행
docker run -d -p 3000:3000 --name [server-name] --restart always \
  -e JWT_KEY=$JWT_KEY \
  ***.dkr.ecr.ap-northeast-2.amazonaws.com/$REPO_NAME:$LATEST_TAG

```

## CI/CD workflow

ci/cd workflow는 아래와 같이 작성했다. 이 과정에서 docker 이미지 캐시하는 부분에서 시간이 너무 오래걸려 일단은 이 부분을 제외했다.
나중에 cache key가 너무 변경되지 않도록 workflow를 수정할 예정이다.

배포하는 플로우를 보면 다음과 같이 구성되어 있다.

1. ssh 접속 (Ec2)
2. aws cli configuration
3. ecr 로그인
4. ecr 이미지 pull
5. docker image 실행

```
name: Deploy

on:
  push:
    branches: [development]

env:
  AWS_REGION: ap-northeast-2
  ECR_REPOSITORY: threads
  IMAGE_TAG: ${{ github.sha }}
  NAME: threads
  SECRET_MANAGER_ID: dev-secret

jobs:
  build:
    name: Build
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v2

      - name: Setup docker buildx
        id: buildx
        uses: docker/setup-buildx-action@v1

      - name: Cache docker layers
        uses: actions/cache@v2
        with:
          path: /tmp/.buildx-cache
          key: ${{ runner.os }}-buildx-${{ env.IMAGE_TAG }}
          restore-keys: |
            ${{ runner.os }}-buildx-

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: ecr-login
        run: |
          aws ecr get-login-password --region ${{ env.AWS_REGION }} | docker login --username AWS --password-stdin ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com

      - name: Build Docker image
        run: |
          docker build -t ${{ env.ECR_REPOSITORY }}:${{ env.IMAGE_TAG }} .

      - name: Tag and Push Docker image to ECR
        run: |
          docker tag ${{ env.ECR_REPOSITORY }}:${{ env.IMAGE_TAG }} ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/${{ env.ECR_REPOSITORY }}:${{ env.IMAGE_TAG }}
          docker push ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/${{ env.ECR_REPOSITORY }}:${{ env.IMAGE_TAG }}

  deploy:
    needs: build
    name: Deploy
    runs-on: ubuntu-latest
    steps:
      - name: SSH to EC2 and Deploy
        uses: appleboy/ssh-action@v0.1.7
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USER }}
          key: ${{ secrets.EC2_SSH_KEY }}
          port: ${{ secrets.EC2_PORT }}
          script: |
            # Ensure AWS CLI and Docker are installed
            if ! command -v aws &> /dev/null; then
              sudo apt-get update
              sudo apt-get install -y awscli
            fi
            if ! command -v docker &> /dev/null; then
              curl -fsSL https://get.docker.com -o get-docker.sh
              chmod +x get-docker.sh
              sudo sh get-docker.sh
            fi

            # Configure AWS CLI with access `keys
            aws configure set aws_access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
            aws configure set aws_secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}
            aws configure set default.region ${{ env.AWS_REGION }}

            # Fetch secrets from AWS Secrets Manager
            secret_json=$(aws secretsmanager get-secret-value --secret-id ${{ env.SECRET_MANAGER_ID }} --query SecretString --output text)

            # Extract individual secrets and export them as environment variables


            # Login to ECR
            aws ecr get-login-password --region ${{ env.AWS_REGION }} | docker login --username AWS --password-stdin ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com

            # Stop and remove the existing container
            if docker inspect threads &> /dev/null; then
              docker stop threads
              docker rm threads
            fi

            # Pull the new image and run the container with environment variables
            docker pull ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/${{ env.ECR_REPOSITORY }}:${{ env.IMAGE_TAG }}
            docker run -d -p 3000:3000 --name threads --restart always \
              ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com/${{ env.ECR_REPOSITORY }}:${{ env.IMAGE_TAG }}

```

이 과정에서 조금 아쉬웠던 건, ec2 ssh 접속할 때 public ip를 사용중인데 ec2 서버가 죽었을 땐 secret 값을 계속 변경해줘야 하므로 이 부분은 ec2서버가 죽어도 secret 값을 수정하지 않는 방향으로 수정할 예정이다.

## 앞으로 남은 과제

내가 CI/CD를 적용하기 위해 기존 인프라를 일단은 위처럼 대체하긴 했지만, Best는 아니었다고 생각한다. 로드밸런서도 너무 많이 구성했다는 생각도 들고 더 좋은 방법이 있었을 것이라 생각한다.

추가로 CI/CD 를 도입하는 과정에서, github action 에는 ssh 접속을 하기 위해 ec2의 public ip를 github secret 에 넣어놨는데 관리포인트를 늘린게 아닌가 하는 아쉬움도 남는다. 물론 ec2 서버가 죽을 때를 생각해서 EIP 생성 후, lambda를 통해 새로 생성되는 ec2에 할당하거나 하는 생각도 하긴 했지만 일단은 이렇게 두었다.

더 빠르게 배포하고 더 좋은 CI/CD 플로우를 만들기 위해 더 노력하겠다는 다짐을 하고 글을 마친다.
