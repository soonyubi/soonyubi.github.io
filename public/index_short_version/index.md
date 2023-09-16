# Index (간략버전)


## 개요
> index는 어떤 column에 대해 특정 기준을 적용해서 `정렬`함으로써 데이터를 더 빠르게 찾을 수 있는 방법을 말한다.

이와 같은 방법을 적용하는 사례는 사실 우리 주변에도 존재한다. 예를 들어 주민등록번호, 살고있는 주소, 학번 등 이미 인덱스를 활용하고 있고 그렇게 어려운 개념은 아니다. 

개발면접에서 인덱스를 물어보는 질문은 꽤나 많다. 단순히 index에 대한 개념을 묻는 질문은 아니라고 생각한다. DB에서 인덱스를 어떻게 사용하고, 어떤 자료구조를 이용하고, 인덱스를 사용함으로써 어떻게 최적화를 이뤄낼 수 있는지에 대한 경험이나 지식을 물어보는 것이라 생각한다.

## clustered index vs non-clustered index

### clustered index 
pk 값이 비슷한 레코드끼리 묶어서 저장되고 사용하는 것을 말한다. pk를 기준으로 쿼리를 수행하기 때문에, pk 선택에 있어서 신중히 해야 한다. 가급적 거의 업데이트 되지 않거나, 전혀 업데이트 되지 않는 column을 선택해야 한다. 순서를 유지하고 범위 검색에 유리하고 데이터가 많아질수록 삽입 성능이 떨어지는 특징이 있다.<br><br>

{{< image src="/images/clustered_index.png" caption="clustered index" width="100%" >}}

### non-clustered index
`secondaryt index`라고 불리우는데, 테이블 데이터와 함께 테이블에 저장되는 것이 아니라 별도의 저장소에 저장되고 그 저장된 값은 포인터로써 실제 저장된 데이터를 가르킨다. 순서를 유지하지 않고, 한 테이블에 여러개 존재할 수 있고, index를 저장할 추가적인 공간(약 10%)이 필요하다는 특징이 있다. <br><br>


{{< image src="/images/secondary_index.png" caption="secondary index" width="100%" >}}

## Mysql 최적화와 관련해서 생각해볼 것들

이 부분은 직접 내가 경험해보지 않았고, 나중에 최적화가 필요할 때 찾아볼 개념들이다. [여기](https://crystalcube.co.kr/163)에서는 인덱스를 활용해서 몇몇 실험을 한 결과를 볼 수 있다. 다음을 이용해서 최적화를 경험할 경우 별도의 포스트에 해당 경험을 공유하려고 한다.

- `Explain` keyword 
- B-tree, page in InnoDB
- `Cardinality` : index 선택 기준에 대한 지표
- `Composite Key`
- `innoDB_buffer_pool_size`
- `log_throttle_queries_not_using_indexes`

## Elastic Search for Indexing

토이 프로젝트에 게시글을 더 빠르게 검색하기 위해서 postgre DB 앞단에 Elastic Search를 별도로 사용중이다. elastic search는 restful api로 서비스를 이용할 수 있다.

### /GET : 게시글 요청  
검색란에 text를 써서 요청하면, 다음 함수가 실행된다. 

```javascript
async search(text: string){
  const {body} = await this.elasticSearchService.search<PostSearchResult>({
    index : 'index',
    body : {
      query : {
        multi_match : {
          query : text,
          fields : ['title','content']
        }
      }
    }
  });
}
```


### /POST : 게시글 생성
```javascript
async indexPost(post: Post) {
    return this.elasticsearchService.index<PostSearchResult, PostSearchBody>({
      index: this.index,
      body: {
        id: post.id,
        title: post.title,
        content: post.content,
        authorId: post.author.id
      }
    })
  }
```

### /PATCH : 게시글 업데이트

```javascript
async update(post: Post) {
    const newBody: PostSearchBody = {
      id: post.id,
      title: post.title,
      content: post.content,
      authorId: post.author.id
    }
 
    const script = Object.entries(newBody).reduce((result, [key, value]) => {
      return `${result} ctx._source.${key}='${value}';`;
    }, '');
 
    return this.elasticsearchService.updateByQuery({
      index: this.index,
      body: {
        query: {
          match: {
            id: post.id,
          }
        },
        script: {
          inline: script
        }
      }
    })
  }
```

업데이트시에 주목할 부분은, 관련 script를 직접 구성해줘야 한다.

### /DELETE : 게시글 삭제

```javascript
async remove(postId: number) {
    this.elasticsearchService.deleteByQuery({
      index: this.index,
      body: {
        query: {
          match: {
            id: postId,
          }
        }
      }
    })
  }
```

## 참고자료
[링크1](https://www.youtube.com/watch?v=NkZ6r6z2pBg)

[링크2](https://gwang920.github.io/database/clusterednonclustered/)

[링크3](https://crystalcube.co.kr/163)

[링크4](http://jidum.com/jidums/view.do?jidumId=183)

[링크5](https://choseongho93.tistory.com/231)

[링크6](https://hoing.io/archives/5960#InnoDB_-_PK_-_Clustered_Index)

[링크7](https://yurimkoo.github.io/db/2020/03/14/db-index.html)
