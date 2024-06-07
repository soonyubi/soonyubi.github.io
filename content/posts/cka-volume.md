---
title: "[CKA] volume"
subtitle: ""
date: 2024-05-27T15:37:16+09:00
lastmod: 2024-05-27T15:37:16+09:00
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

### persistent volume and persistent volume claim

pod에서 실행중인 컨테이너가 /log/app.log 파일에 로그를 저장한다고 가정해보자. 만약 pod가 재시작 될 경우, 해당 log 파일은 사라지기 때문에, host pc에 volume을 mount 하여 사용하여야 한다.

이렇게 지정하게 되면, 어느 pod가 어떤 volume을 마운트했는지 관리하기 어려워지므로, 쿠버네티스에서는 스토리지 관리 시스템인 persistent volume 과 pv의 추상화 레벨인 persistent volume claim을 지원한다.

persistent volume은 실제 스토리지를 의미하고, pvc는 pod가 스토리지를 사용할 수 있게 스토리지를 요청하는 것을 의미한다.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-example
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: "/mnt/data"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-example
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: Pod
metadata:
  name: pod-example
spec:
  containers:
    - name: container-example
      image: nginx
      volumeMounts:
        - mountPath: "/usr/share/nginx/html"
          name: storage
  volumes:
    - name: storage
      persistentVolumeClaim:
        claimName: pvc-example
```

위와 같이 pv, pvc를 정의하게 되면, pv 는 노드 내에 스토리지를 생성하고, pvc는 10gi를 만족할 수 있는 pv 리소스를 찾게 된다.
pvc는 pod 생성 시 volume mount 로 연결된다. 이 때 위 명세서를 보면, pv는 1gi 를 생성했고, pvc는 10gi를 요청하므로 적절한 pv 를 찾을 수 없어 pod는 pending 상태로 남아있을 것이다.

또한 위에 정의되었듯이, pv와 pvc의 `accessMode`도 서로 일치해야 한다.

- reclaim policy
  위 명세서를 보면 pv의 reclaim policy는 `retain`으로 할당되어 있다. 만약 pvc가 삭제되어도 pv와 그 안의 데이터는 보존됨을 의미한다. 이때 pv는 `released` 상태로 변경되고 pv를 다시 재사용하기 위해선, pv 내부 데이터를 삭제하거나 수동으로 pvc를 바인딩해줘야 한다.

  만약 pvc가 pod에 할당되어 있는 상태라면, kubectl delete pvc 명령어를 사용했을 경우, hang이 걸릴 수 있다.

### storage classes

스토리지를 동적으로 provisioning 할 때 사용하는 리소스이다. 예를 들어 AWS EBS를 사용하거나 GCE 를 사용할 때 사용한다.
pvc를 이용해서도 동적 provision을 할 수 있는데 `storageClassName`에 특정 이름을 적게 되면, kubernetes 관리자는 자동으로 pv를 생성하고 이를 할당한다.

- provisioner: kubernetes.io/no-provisioner : local storage를 생성하는 방법으로, 동적으로 스토리지를 생성하지 않고 수동으로 스토리지를 생성한다. 이 값을 사용하는 경우 `volumeBindingMode: WaitForFirstConsumer` 로 설정되는데 이는 PVC가 실제로 사용되기 전까지 PV에 할당하는 것을 지연하는 것을 의미한다.
