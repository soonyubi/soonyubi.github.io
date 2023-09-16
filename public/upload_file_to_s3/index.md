# public/private s3 를 생성하고 파일 업로드 기능 구현


## 개요
이번 포스팅에서는 public/private s3를 생성하고 각각 어떻게 접근하는지, 그리고 코드로서는 어떻게 구현하는지에 대해 알아보겠다.

## IAM User 생성
AWS Root user는 모든 서비스에 대한 권한을 가지고 있기 때문에 `at least privilege` 의 원칙에 어긋난다. 따라서 `IAM User` 를 생성해 S3에 대한 접근 권한만을 부여하도록 한다. 

## Public/Private S3 생성

#### Public
모든 퍼블릭 엑세스 차단 체크박스를 해제하고 생성한다.
추가로, `Bucket Policy` 에 다음과 같이 작성하여, url을 통해 bucket에 접근할 수 있도록 한다. 

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AddPerm",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "[생성한 S3 ARN]/*"
        }
    ]
}
```

#### private
모든 퍼블릭 액세스 차단 체크박스를 체크하고 생성한다. 

## 구현 

### config
```javascript
import { ConfigService } from '@nestjs/config';
import { config } from 'aws-sdk';

const configService = app.get(ConfigService);
config.update({
  accessKeyId: configService.get('AWS_ACCESS_KEY_ID'),
  secretAccessKey: configService.get('AWS_SECRET_ACCESS_KEY'),
  region: configService.get('AWS_REGION'),
});
```

### 엔티티

#### private
```javascript
@Entity()
export class PrivateFile{
    @PrimaryGeneratedColumn()
    public id: number;

    @Column()
    public key: string;

    @ManyToOne(() => User, (owner: User) => owner.files)
    public owner?: User;
}
```

#### public
```javascript
@Entity()
class PublicFile {
    @PrimaryGeneratedColumn()
    public id: number;
    
    @Column()
    public url: string;
    
    @Column()
    public key: string;
}
```

### 업로드 기능 
거의 차이는 없지만 주목할 부분은, public에서는 해당 entity에 `location을` 저장하고 private에서는 해당 객체의 소유자를 저장하는 모습을 볼 수 있다.

#### private
```javascript
async uploadPrivateFiles(dataBuffer : Buffer, ownerId : number, filename : string){
        const s3 = new S3();
        const uploadResult = await s3.upload({
            Bucket : this.configService.get("AWS_PRIVATE_BUCKET_NAME"),
            Body: dataBuffer,
            Key: `${uuid()}-${filename}`
        }).promise();

        const newFile = await this.privateFilesRepository.create({
            key: uploadResult.Key,
            owner : {
                id : ownerId
            }
        });

        await this.privateFilesRepository.save(newFile);

        return newFile;
    }

```

#### public
```javascript
async uploadPublicFile(dataBuffer : Buffer, filename : string){
        const s3 = new S3();
        const uploadResult = await s3.upload({
            Bucket: this.configService.get("AWS_PUBLIC_BUCKET_NAME"),
            Body: dataBuffer,
            Key:`${uuid()}-${filename}`
        }).promise();

        const newFile = this.publicFilesRepository.create({
            key : uploadResult.Key,
            url: uploadResult.Location
        });

        await this.publicFilesRepository.save(newFile);

        return newFile;
        
    }
```

{{< admonition note "promise()를 붙여주는 이유" >}}
aws-sdk는 node.js 환경에서 비동기적으로 작동하며, callback 함수를 사용하는 것보다 더 가독성 있게 코드를 만들 수 있기 때문
{{< /admonition >}}



### Fetch 기능
public s3의 업로드된 파일을 fetch하는 과정은 필요없다. 왜냐하면 url로 해당 객체에 대한 접근을 할 수 있기 때문이다. 
만약 파일이 브라우저에서 해석할 수 있는 포맷이라면 바로 보여주고, 그렇지 않다면 다운로드를 하게 된다. 

#### private
1. presigned url 을 생성해서 가져오기 
```javascript
public async generatePresignedUrl(key: string) {
    const s3 = new S3();

    return s3.getSignedUrlPromise('getObject', {
      Bucket: this.configService.get('AWS_PRIVATE_BUCKET_NAME'),
      Key: key
    })
  }
```
<br>
2. readable stream을 생성하기 <br>
AWS SDK에서 얻은 readable stream을 사용하여 파일을 다운로드하지 않고도 서버의 메모리를 절약할 수 있다는 것을 설명하고 있습니다. '스트림'은 데이터를 일정한 크기의 '조각'으로 나누어 전송하는 방식을 말하며, 이는 대용량의 데이터를 처리할 때 효율적인 방법입니다. '파이프'는 두 개의 스트림을 연결하여 데이터의 이동을 용이하게 하는 방식을 말합니다. 따라서, 이 문장은 스트림을 사용하여 데이터를 처리하고 서버의 메모리를 절약하는 방법을 제시하고 있습니다.


<br>

```javascript
public async getPrivateFile(fileId: number) {
    const s3 = new S3();

    const fileInfo = await this.privateFilesRepository.findOne({ id: fileId }, { relations: ['owner']});
    if (fileInfo) {
      const stream = await s3.getObject({
        Bucket: this.configService.get('AWS_PRIVATE_BUCKET_NAME'),
        Key: fileInfo.key
      })
        .createReadStream();
      return {
        stream,
        info: fileInfo,
      }
    }
    throw new NotFoundException();
  }
```

### 컨트롤러 
`@UseInterceptors(FileInterceptor('file'))` 


```javascript
@Delete('avatar')
  @UseGuards(JwtAuthenticationGuard)
  async deleteAvatar(@Req() request: RequestWithUser) {
    return this.usersService.deleteAvatar(request.user.id);
  }

  @Get('files')
  @UseGuards(JwtAuthenticationGuard)
  async getAllPrivateFiles(@Req() request: RequestWithUser) {
    return this.usersService.getAllPrivateFiles(request.user.id);
  }

  @Post('files')
  @UseGuards(JwtAuthenticationGuard)
  @UseInterceptors(FileInterceptor('file'))
  async addPrivateFile(@Req() request: RequestWithUser, @UploadedFile() file: Express.Multer.File) {
    return this.usersService.addPrivateFile(request.user.id, file.buffer, file.originalname);
  }

  @Get('files/:id')
  @UseGuards(JwtAuthenticationGuard)
  async getPrivateFile(
    @Req() request: RequestWithUser,
    @Param() { id }: FindOneParams,
    @Res() res: Response
  ) {
    const file = await this.usersService.getPrivateFile(request.user.id, Number(id));
    file.stream.pipe(res)
  }
```
## 마무리
