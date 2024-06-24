---
title: "[Spring] 블로그 프로젝트 - DB ACCESS"
subtitle: ""
date: 2024-06-12T16:38:45+09:00
lastmod: 2024-06-12T16:38:45+09:00
draft: false
author: ""
authorLink: ""
description: ""
license: ""
images: []

tags: []
categories: ["spring"]

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

## execute mysql server

```yaml
// docker-compose for execute mysql server in docker
version: '3.8'

services:
  mysql:
    image: mysql:8.0
    container_name: mysql-container
    environment:
      MYSQL_ROOT_PASSWORD: root_password
      MYSQL_DATABASE: blog_db
      MYSQL_USER: user
      MYSQL_PASSWORD: user_password
    ports:
      - "3306:3306"
    volumes:
      - mysql-data:/var/lib/mysql

volumes:
  mysql-data:
```

## configure db connection

### install dependency

```
dependencies {
  ...
	implementation("org.springframework.boot:spring-boot-starter-data-jpa")
	implementation("mysql:mysql-connector-java:8.0.32")
}
```

### set configuration to access mysql server

```properties
// application.properties

spring.datasource.url=jdbc:mysql://localhost:3306/blog_db
spring.datasource.username=user
spring.datasource.password=user_password
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.format_sql=true
```

blog application을 실행시켰을 때 정상적으로 서버가 띄어지면, db connection에 성공한 것으로 생각할 수 있다.

## define entity

### PrimaryKeyEntity

모든 entity에 대해 UUID를 이용해 PK를 관리하고, `Persistable` interface를 구현해서 isNew 메서드를 커스텀화하고 isNew property에 대한 상태관리를 좀 더 구체화하기 위해 PrimaryKeyEntity를 정의하고자 한다.

```kotlin

@MappedSuperclass
abstract class PrimaryKeyEntity : Persistable<UUID> {
    @Id
    @Column(columnDefinition = "uuid")
    private val id : UUID = UUID.nameUUIDFromBytes(ULID().nextValue().toString().toByteArray())

    @Transient
    private var _isNew : Boolean = true

    override fun getId() = id

    override fun isNew() = _isNew

    override fun equals(other: Any?): Boolean {
        if(other == null){
            return false
        }

        if(other !is HibernateProxy && this::class != other::class){
            return false
        }

        return id == getIdentifier(other)
    }

    private fun getIdentifier(obj: Any) : Serializable {
        return if(obj is HibernateProxy){
            obj.hibernateLazyInitializer.identifier as Serializable
        }else{
            (obj as PrimaryKeyEntity).id
        }
    }

    override fun hashCode(): Int = Objects.hashCode(id)

    @PostPersist
    @PostLoad
    protected fun load(){
        _isNew = false
    }
}
```

- `@MappedSuperclass`
  JPA 에서 사용하는 annotation으로, 이 annotation이 정의된 클래스를 상속받는 모든 클래스들이 PrimaryKeyEntity에 정의된 매핑 정보를 상속받을 수 있도록 해주는 annotation이다.

- `@Transient`
  @Transient annotation은 객체레벨에서만 필드를 쓰도록 허용하는 것이며, 실제 엔티티에 해당 필드를 생성하지 않게 하는 annotation이다.

- `@PostPersist`
  entity가 persist 된 후에 해당 annotation이 할당된 함수를 실행하라는 의미이다.

- `@PostLoad`
  entity가 데이터베이스에서 로드되었을 때(조회) 해당 annotation이 할당된 함수를 실행하라는 의미이다.

- equals()/hashCode()를 overriding
  위 2개 함수를 오버라이딩하여, 동등성 비교에 대한 로직을 커스텀화하였다.

### BlogEntity

```kotlin
@Entity
class PostEntity(
    title: String,
    content: String,
    information: PostInformation
) : PrimaryKeyEntity() {
    @Column(nullable = false)
    var createdAt : LocalDateTime = LocalDateTime.now()
        protected set

    @Column(nullable = false)
    var title: String = title
        protected set

    @Column(nullable = false)
    var content: String = content
        protected set

    @Embedded
    var information : PostInformation = information
        protected set

    fun update(data: PostUpdateData){
        this.title=data.title
        this.information=data.information
        this.content=data.content
    }
}

@Embeddable
data class PostInformation(
    @Column
    val link: String?,

    @Column(nullable = false)
    val rank: Int
)

data class PostUpdateData(
    val title:String,
    val content: String,
    val information: PostInformation
)
```

여기서 information 필드를 보면, `@Embeddable`로 선언된 PostInformation data class 를 `@Embedded` 한 것을 볼 수 있다. 이렇게 함으로써 엔티티 설계를 좀 더 객체지향적으로 설계할 수 있게 된다. 물리적으로는 PostInformation 에 있는 필드들은 Post entity의 필드로써 들어가게 된다.

## create simple api to create/get/update/delete blog

### repository

```kotlin
interface PostRepository : JpaRepository<Post, String>{}
```

### service

```kotlin

@Service
class PostService(
    private val postRepository: PostRepository
){
    fun getAllPosts() : List<Post> = this.postRepository.findAll()


    fun getById(id: String) : Post? = this.postRepository.findById(id).orElseThrow()

    @Transactional
    fun create(data : PostCreationPayload) {
        this.postRepository.save(data.toEntity())
    }

    @Transactional
    fun update(id:String, data : PostUpdatePayload){
        getById(id)?.apply { update(data.toData()) }
    }

    @Transactional
    fun delete(id: String){
        postRepository.deleteById(id)
    }
}
```

### controller

```kotlin

@RestController
@RequestMapping("/api/posts")
class PostController (
    private val postService: PostService
){
    @GetMapping
    fun getAllPosts() : List<Post> = postService.getAllPosts()

    @GetMapping("/{id}")
    fun getPostById(@PathVariable id : String) : ResponseEntity<Post> {
        val post = postService.getById(id)
        return if(post!=null){
            ResponseEntity.ok(post)
        }else{
            ResponseEntity.notFound().build()
        }
    }

    @PostMapping
    fun createPost(@RequestBody data : PostCreationPayload){
        postService.create(data)
    }

    @PatchMapping("/{id}")
    fun updatePost(@PathVariable id: String,@RequestBody data : PostUpdatePayload){
        postService.update(id, data)
    }

    @DeleteMapping("/{id}")
    fun deletePost(@PathVariable id : String){
        postService.delete(id)
    }
}
```
