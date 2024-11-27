---
title: Hextra Theme Guide
type: docs
prev: /
next: docs/folder/
---

# Hextra를 활용한 파일 구성 가이드

Hextra는 Hugo를 기반으로 동작하는 정적 사이트 생성기로, `content` 디렉토리 구조를 기반으로 웹사이트의 출력 구조를 결정합니다. 이 문서에서는 Hextra의 파일 구성 및 레이아웃, 내비게이션 설정에 대해 자세히 설명합니다.

---

## 디렉토리와 URL 매핑

Hextra에서 `content` 디렉토리는 사이트 구조의 핵심입니다. `content` 아래의 디렉토리와 파일 구조는 웹사이트의 URL 구조에 그대로 매핑됩니다.

### 예시 디렉토리 구조

```
content/
└── docs/
    ├── guide/
    │ ├── organize-files.md
    │ └── \_index.md
    └── overview.md
```

### 매핑된 URL

- `content/docs/overview.md` → `/docs/overview/`
- `content/docs/guide/organize-files.md` → `/docs/guide/organize-files/`

---

## 레이아웃 설정

Hextra는 다음과 같은 기본 레이아웃을 제공합니다:

- **Docs**: 문서에 적합한 구조화된 레이아웃
- **Blog**: 블로그 게시물 스타일
- **Default**: 일반 페이지에 사용하는 기본 레이아웃

### 특정 레이아웃 설정

특정 디렉토리의 레이아웃을 지정하려면 해당 디렉토리의 `_index.md` 파일에서 `type`을 설정합니다.

```yaml
---
title: "Guide"
type: "docs"
---
```

위 예시는 /docs/guide/ 디렉토리에서 docs 레이아웃을 사용하도록 설정합니다.

---

## 사이드바 내비게이션

사이드바는 content 디렉토리 구조와 파일 이름의 알파벳 순서에 따라 자동 생성됩니다. 순서를 사용자 정의하려면 파일의 front matter에 weight 파라미터를 추가하세요.

순서 조정 예시

content/docs/guide/ 디렉토리 구조:

```
content/
  └── docs/
      └── guide/
          ├── _index.md (weight: 1)
          ├── basics.md (weight: 2)
          └── advanced.md (weight: 3)
```

---

## 브레드크럼 내비게이션

브레드크럼(Breadcrumb) 내비게이션은 사용자가 현재 위치한 페이지가 사이트 전체 구조에서 어디에 있는지 보여줍니다. /content 디렉토리 구조를 기반으로 자동 생성됩니다.

브레드크럼 표시 예시

예를 들어, /docs/guide/organize-files/ 페이지는 다음과 같은 브레드크럼을 표시합니다:
linkTitle을 사용하여 표시되는 텍스트를 변경할 수 있습니다:

```yaml
---
title: "Organize Files"
linkTitle: "File Organization"
---
```

---

## 이미지 추가 방법

Hextra는 이미지를 Markdown 파일과 함께 관리하거나, static/ 디렉토리를 활용해 모든 페이지에서 접근할 수 있도록 지원합니다.

### 페이지 번들을 활용한 이미지 관리

Markdown 파일과 이미지를 같은 디렉토리에 저장:

```
content/docs/my-page/
  ├── index.md
  └── image.png

참조할 경우 `![설명 텍스트](image.png)`
```

### static/ 디렉토리 활용

static/images/에 이미지 저장:
`static/images/my-image.png`
`![설명 텍스트](/images/my-image.png)`
