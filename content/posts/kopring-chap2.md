---
title: "[Spring] 블로그 프로젝트 - Relation"
subtitle: ""
date: 2024-06-14T14:36:20+09:00
lastmod: 2024-06-14T14:36:20+09:00
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

## Define User Entity(OneToMany, ManyToOne)

### User entity

```kotlin
@Entity
@Table(name = "`user`")
class User(
    name: String
) : PrimaryKeyEntity(){
    @Column(nullable = false, unique = true)
    var name : String = name
        protected set

    @OneToMany(fetch = FetchType.LAZY, cascade = [CascadeType.ALL], mappedBy = "writer")
    protected var mutablePosts : MutableList<Post> = mutableListOf()

    val posts: List<Post> get() = mutablePosts.toList()

    fun writePost(post: Post){
        mutablePosts.add(post)
    }
}
```

- `fetch = FetchType.LAZY `
  이 옵션은, 유저를 조회할 때 모든 post가 딸려오는 것이 아니라, user.posts를 조회할 때 posts entity들이 load되도록 하는 것을 의미한다.
- `cascade = [CascadeType.ALL]`
  이 옵션은, 유저를 삭제, 수정, 저장할 때 post entity도 같이 삭제, 수정, 저장이 되도록 하는 옵션이다.
- `mappedBy = "writer"`
  양방향 관계에서 진짜 주인은, writer 필드가 선언된 post Entity임을 의미한다.

### Modify Post Entity

```kotlin
// Post.kt

@ManyToOne(fetch = FetchType.LAZY, optional = false)
@JoinColumn(nullable = false)
var writer : User = user
  protected set

init {
  writer.writePost(this)
}
```

- `@JoinColumn`
  JoinColumn은 ManyToOne 관계에서 외래키를 지정하는데 사용한다. 주로 다대일 관계에서 "다"쪽에 선언한다.
  nullable=false 로 선언한 이유는 해당 컬럼이 null 값을 가질 수 없음을 의미한다. 데이터베이스 수준에서 필수값을 강제한다.
- `@ManyToOne(fetch = FetchType.LAZY, optional = false)`
  optional=false 값의 의미는, 엔티티 수준에서 해당 필드는 null이 될 수 없음을 의미한다. 고로, Post Entity 를 초기화할 때 writer의 값은 강제라고 생각할 수 있다.
- init block
  Post class가 인스턴스화 할 때 실행되는 블록이다. Post가 인스턴스화할 때 writer 객체에도 값을 추가하므로 데이터 무결성을 보장할 수 있다.

## Define Tag Entity (ManyToMany)

### Tag Entity

```kotlin
@Entity
@Table(name="tag")
class Tag(
    key : String,
    value: String
) : PrimaryKeyEntity(){
    @Column(nullable = false, name = "`key`")
    var key : String = key
        protected set

    @Column(nullable = false, name = "`value`")
    var value : String = value
        protected set
}
```

### Modify Post Entity

```kotlin
  @ManyToMany(fetch = FetchType.LAZY, cascade = [CascadeType.PERSIST, CascadeType.MERGE])
    @JoinTable(
        name = "board_tag_assoc",
        joinColumns = [JoinColumn(name = "post_id")],
        inverseJoinColumns = [JoinColumn(name = "tag_id")]
    )
    protected var mutableTags: MutableSet<Tag> = tags.toMutableSet()
    val tags: Set<Tag> get() = mutableTags.toSet()
```

- `@ManyToMany(fetch = FetchType.LAZY, cascade = [CascadeType.PERSIST, CascadeType.MERGE])`
  post entity가 저장될 때, 그의 자식 엔티티인 Tag 엔티티에도 값을 저장하도록 CASCADE 옵션을 사용하였다.
- @JoinTable(
  name = "board_tag_assoc",
  joinColumns = [JoinColumn(name = "post_id")],
  inverseJoinColumns = [JoinColumn(name = "tag_id")]
  )
  다대다 관계에서 associate 테이블을 정의하기 위해 사용한 annotation이다.
  JPA가 자동으로 해당 엔티티를 선언해주며, Post entity가 수정/저장될 때 tag entity를 수정하거나 저장하고 JPA가 관리하고 있는 associate 테이블에도 값을 추가해준다.

## Define Comment

```
@Embeddable
data class Comment(
    @Column(name = "content", length = 3000)
    private var _content : String,

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "writer_id")
    private var _writer : User
){
    val content : String get() = _content
    val writer : User get() = _writer
}

// post.kt
@ElementCollection
    @CollectionTable(name = "post_comment")
    private val mutableComments : MutableList<Comment> = mutableListOf()
    val comments: List<Comment> get() = mutableComments.toList()

```
