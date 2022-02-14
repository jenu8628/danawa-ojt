

# 01/04

## docker란?

> 리눅스 운영체제에서 지원하는 다양한 기능을 사용해 컨테이너(하나의 프로세스)를 실행하기 위한 별도의 환경(파일 시스템)을 준비하고, 리눅스 네임스페이스와 다양한 커널 기능을 조합해 프로세스를 특별하게 실행시켜 준다.
>
> 이는 가상머신과 같이 하드웨어를 가상화하는 것이 아니라, 운영체제 상에서 지원하는 방법을 통해 하나의 프로세스(컨테이너)를 실행하기 위한 별도의 환경을 구축하는 일을 지원하고, 도커는 바로 프로세스를 격리시켜 실행해주는 도구이다.



## 컨테이너

> 개별 소프트웨어의 실행에 필요한 실행환경을 독립적으로 운용할 수 있도록 기반환경 또는 다른 실행환경과의 간섭을 막고 **실행의 독립성을 확보해주는 운영체계 수준의 격리 기술**
>
> 컨테이너는 애플리케이션을 실제 구동 환경으로부터 **추상화**할 수 있는 논리 패키징 매커니즘을 제공한다.

- 소프트웨어의 실행을 위해선 OS와 Library를 포함, 소프트웨어가 필요로 하는 파일 등으로 구성된 실행환경이 필요한데, 하나의 시스템 위에서 둘 이상의 소프트웨어를 동시에 실행하려고 한다면 문제가 발생할 수 있다.
- ex) software A와 B가 동일한 라이브러리를 사용하지만 서로 다른 버전을 필요로 하는 경우나 두 소프트웨어의 운영 체제가 다를 경우 등 다양한 경우에서 문제가 발생할 수 있다.
-  이런 상황에서 가장 간단한 해결책은 두 소프트웨어를 위한 시스템을 각각 준비하는 것인데 시스템을 각각 준비할 경우 비용의 문제가 발생하게 됨(10개의 소프트웨어일 경우 10개의 시스템이 필요)
- 이러한 문제를 효율적으로 해결한 것이 바로 컨테이너



## 이미지

> 컨테이너 실행에 필요한 파일과 설정값등을 포함하고 있는 것으로 상태값을 가지지 않고 변하지 않는다.

- 컨테이너는 이미지를 실행한 상태라고 볼 수 있고 추가되거나 변하는 값은 컨테이너에 저장된다.
- 같은 이미지에서 여러개의 컨테이너를 생성할 수 있고 컨테이너의 상태가 바뀌거나 컨테이너가 삭제되더라도 이미지는 변하지 않고 그대로 남아있다.

### 1. Doker Layer

- 도커 이미지는 컨테이너를 실행하기 위한 모든 정보를 가지고 있기 때문에 보통 용량이 수백메가MB에 이른다. 처음 이미지를 다운받을 땐 크게 부담이 되지 않지만 기존 이미지에 파일 하나 추가했다고 수백메가를 다시 다운받는다면 매우 비효율적이다
- 이러한 문제를 해결하기 위한 개념이 Layer이다.
- 유니온 파일 시스템을 이용하여 여러개의 레이어를 하나의 파일시스템으로 사용할 수 있게 해준다
- 이미지는 여러개의 읽기 전용 read only레이어로 구성되고 파일이 추가되거나 수정되면 새로운 레이어가 생성된다.
- ex) ubuntu 이미지가 `A` + `B` + `C`의 집합이라면, ubuntu 이미지를 베이스로 만든 nginx 이미지는 `A` + `B` + `C` + `nginx`가 됩니다.



## 컨테이너 실행하기

도커를 실행하는 명령어는 다음과 같습니다.

```
doker run [options] IMAGE[:TAG|@DIGESET] [COMMAND] [ARG...]
```

다음은 자주 사용하는 옵션들입니다.

| 옵션  | 설명                                                         |
| :---- | :----------------------------------------------------------- |
| -d    | detached mode 흔히 말하는 백그라운드 모드                    |
| -p    | 호스트와 컨테이너의 포트를 연결 (포워딩)                     |
| -v    | 호스트와 컨테이너의 디렉토리를 연결 (마운트)                 |
| -e    | 컨테이너 내에서 사용할 환경변수 설정                         |
| -name | 컨테이너 이름 설정                                           |
| -rm   | 프로세스 종료시 컨테이너 자동 제거                           |
| -it   | -i와 -t를 동시에 사용한 것으로 터미널 입력을 위한 옵션       |
| -link | 컨테이너 연결 [컨테이너명:별칭]                              |
| -i    | interacive라는 뜻. 컨테이너와 상호적으로 주고 받고 하겠다. 입력에 대한 출력을 나타내는 말 |
| -t    | tty(터미널과 동일한 의미)를 사용하겠다. 즉 터미널과 비슷한 환경을 조성해줌. |



## 도커 기본 명령어

##### 컨테이너 목록 확인하기 (ps)

```
docker ps [OPTIONS]
```

##### 컨테이너 중지하기 (stop)

```
docker stop [OPTIONS] CONTAINER [CONTAINER...]
```

- 도커 ID의 전체 길이는 64자리 입니다. 하지만 명령어의 인자로 전달할 때는 전부 입력하지 않아도 됩니다. 예를 들어 ID가 abcdefgh…라면 abcd만 입력해도 됩니다. 앞부분이 겹치지 않는다면 1-2자만 입력해도 됩니다.

##### 컨테이너 제거하기 (rm)

- 종료된 컨테이너를 완전히 제거

```
docker rm [OPTIONS] CONTAINER [CONTAINER...]
```

- 중지된 컨테이너 ID를 가져와서 한번에 삭제

```
docker rm -v $(docker ps -a -q -f status=exited)
```

##### 이미지 목록 확인하기 (images)

```
docker images [OPTIONS] [REPOSITORY[:TAG]]
```

##### 이미지 다운로드 하기 (pull)

```
docker pull [OPTIONS] name[:TAG|@DIGEST]
```

##### 이미지 삭제하기 (rmi)

```
docker rmi [OPTIONS] IMAGE [IMAGE...]
```

- images 명령어를 통해 얻은 이미지 목록에서 이미지 ID를 입력하면 삭제가 된다.
- 단, 컨테이너가 실행중인 이미지는 삭제되지 않는다.
- 컨테이너는 이미지들의 레이어를 기반으로 실행중이므로 당연히 삭제할 수 없다.

##### 컨테이너 로그 보기(logs)

- 컨테이너가 정상적으로 동작하는지 확인하는 좋은 방법은 로그를 확인하는 것이다

```
docker logs [OPTIONS] CONTAINER
```

##### 컨테이너 명령어 실행하기 (exec)

- 실행중인 컨테이너에 들어가거나 컨테이너의 파일을 실행하고 싶을 때가 있다.
- run은 새로 컨테이너를 만들어서 실행하고 exec은 실행중인 컨테이너에 명령어로 실행한다.

```
docker exec [OPTIONS] CONTAINER COMMAND [ARG...]
```





```
# 이미지 목록보기
docker images
# containername이란 컨테이너를 image:tag로 생성 및 실행 시킬껀데 바로 터미널을 열것이고 bash를 사용할 것이다.
docker run -it --name containername image:tag /bin/bash

# 컨테이너를 tes:latest이미지를 통해 백그라운드 모드로 실행 포트 10000을 10000으로 맵핑할 것이다. 
docker run -d -p 10000:10000 test:latest

# containername이란 컨테이너를 bash 커맨드로 실행하고 interacive, tty옵션 적용
docker exec -it containername bash

# 이 이미지를 빌드할 것인데 .: 현재위치에 하겠다. 
docker build -t test:latest .
```



## 이미지

- docker pull : 도커 허브 레지스트리에서 로컬로 도커 이미지 내려받기
- docker push : 로컬에 있는 도커 이미지를 도커 허브 레지스트리에 업로드하기
- docker login : 업로드를 하기 전 도커 허브 계정으로 로그인 수행하기
- doker logout : 도커 허브에서 로그아웃 하기
- 

```
# 구문
docker [image] pull [OPTIONS] name[:TAG|@DIGEST]

# ex)
# 배포판 리눅스 이미지인 debian 10.3버전 다운로드
# [:태그] 태그를 포함하지 않으면 자동으로 최신 버전으로 지정된다.
docker pull debian:10.3

# 도커 이미지 세부 정보 조회
docker image inspect [options] IMAGE [IMAGE...]
```

### 이미지 조회

```
docker images
docker image ls
```

- 도커 이미지 세부 정보 조회
  - inspect
  - history

```
docker image inspect [options] IMAGE [IMAGE...]
# 옵션 --format, -f(단축어)
# JSON 형식의 정보 중 지정한 형식의 정보만 출력할 수 있고, {} 중괄호 형식과 대소문자에 유의

# ex) httpd조회
docker image inspect --format="{{ .RepoTags}}" httpd
docker image inspect --format="{{ .Os}}" httpd
docker image inspect --format="{{ .ContainerConfig.Env}}" httpd

docker image history httpd
```



## 컨테이너

### 컨테이너 실행

- 컨테이너 생성 : create
- 시작 : start, 정지 : stop, 일시정지 : pause, 일시정지 해제 : unpause, 재시작 : restart
- 컨테이너 실행 : exec, attach
- 컨테이너 삭제 : rm
- 컨테이너 이름 변경 : rename
- 컨테이너 조회 : ps
  - 실행중인 컨테이너만 조회됨.
  - 전체를 보려면 -a옵션사용
- commit : 생성한 컨테이너의 노드 프로그램 환경과 저장한 소스 코드 그대로 새로운 이미지로 생성할 수 있다.
  - 일반적으로 이미지 생성은 Dockerfile을 작성하여 docker build를 통해 생성한다.

```
# create는 run과 달리 container 내부 접근을 하지 않고 생성(스냅숏)만 수행
# 다음은 수동으로 컨테이너를 제어하는 법이다.
# container-test1의 이름으로 ubuntu:14.04의 이미지를 가지고 생성
docker create -it --name container-test1 ubuntu:14.04
# 현재 status가 created상태로 start를 사용한다
docker start container-test1
# attach 명령은 실행 중인 어플리케이션 컨테이너에 단순한 조회 작업 수행 시 유용
docker attach container-test1
# 나가기
root@975be160d105:/# exit
# 빠져나온 컨테이너가 강제 종료되어 삭제된다.
docker rm container-test1


# 위 작업을 docker run으로 수행하기
docker run -it --name container-test1 ubuntu:14.04 bash
docker rm container-test1

# 한마디로
docker run = [pull] + create + start + [command]
```

### 컨테이너 실습

- SQL 테스트

```
docker pull mysql:5.7
docker run -it mysql:5.7 /bin/bash
/ect/init.d/mysql start
# mysql 시작
mysql -uroot
# mysql db보기
mysql> show databases;
# mysql db생성
mysql> create database dockerdb;
# mysql db보기
mysql> show databases;
# mysql파일 찾아가기
cd / var/lib/mysql
# 조회
ls
# ...dockerdb...
# 나가기
exit
# 컨테이너 이름, ID를 통해 시작 가능
docker start CONTAINERID
# exec를 이용하여 컨테이너에 접근 가능
docker exec -it CONTAINERID bash
# 컨테이너를 종료시키지 않고 빠져나가려면 ctrl + p + q를 동시에 입력


```

- 컨테이너 모니터링 도구 cAdvisor 컨테이너 실행

```
docker run --volume=/:/rootfs:ro --volume=/var/run:/var/run:rw --volume=/sys:/sys:ro --volume=/var/lib/docker/:/var/lib/docker:ro --publish=9559:8080 --detach=true --name=cadvisor google/cadvisor:latest

http://192.168.0.127:9559/ 에 접속하여 /docker를 클릭해 보면 현재 구동 중인 컨테이너들의 ID를 볼 수 있다.
우너하는 컨테이너를 선택하면 컨테이너의 자원 소비량 등을 확인할 수 있다.
```

- 웹서비스 실행을 위한 Nginx 컨테이너 실행

```
# nginx 1.18 이미지 pull
docker pull nginx:1.18
# --name : 컨테이너 이름을 지정
# -d : 컨테이너를 백그라운드에서 실행하고 컨테이너 ID를 출력
# -p : 컨테이너의 80번 포트를 Hist 포트 8001로 오픈
docker run --name webserver1 -d -p 8001:80 nginx:1.18
# 접속 테스트
curl localhost:8001

# 컨테이너의 리소스 사용량 실시간 확인
docker stats webserver1

# 컨테이너의 실행 중인 프로세스 표시
docker top webserver1

# -f : 실시간, -t : 마지막로그까지
# 접근한 웹 브라우저에서 새로고침을 계속 누르면 접근 로그가 실시간으로 기록됨.
docker logs -f webserver1

# 컨테이너 정지
# docker stop -t 10 webserver1 10초 후 정지
docker stop webserver1

# 컨테이너 시작
docker start webserver1

# nginx의 index.html 내용을 변경해서 테스트해보자
# vi index.html
<h1> Hello, Hyun Woo <h1>

#도커 cp 명령을 통해 컨테이너 내부 index.html 파일 경로에 복사
docker cp index.html webserver1:/usr/share/nginx/html/index.html

# 확인
curl localhost:8001
```

- 파이썬 프로그래밍 환경을 컨테이너로 제공

```
# 샘플 코드 작성(로또 프로그램)
vi py_lotto.py
랜덤함수로 로또 구현

파이썬 컨테이너 실행 후 py_lotto.py 샘플 코드 복사
docker run -it -d --name=python_test -p 8900:8900 python
docker cp py_lotto.py python_test:/

파이썬 컨테이너 확인
docker exec -it python_test bash
python
exit()

# 파이썬 관련 도구 확인
pip list

# 파이썬 컨테이너에 설치된 파이썬 모듈을 체크
python -c 'help("modules")'

# 외부에서 파이썬 컨테이너 코드를 실행한다.
docker exec -it python_test python /py_lotto.py
# 로또 게임 횟수를 입력하세요: 5
# 로또번호[1]: [1, 7, 23, 24, 37, 43]
# 로또번호[2]: [4, 10, 11, 18, 27, 29]
# 로또번호[3]: [5, 8, 18, 19, 30, 37]
# 로또번호[4]: [9, 13, 20, 21, 30, 37]
# 로또번호[5]: [11, 15, 23, 31, 41, 43]
```

- node.js 테스트 환경을 위한 컨테이너 실행

```
# node 컨테이너 실행
docker pull node
docker run -d -it -p 8002:8002 --name=nodejs_test node

# 소스 코드 복사 후 실행
docker cp nodejs_test.js nodejs_test:/nodejs_test.js
docker exec -it nodejs_test node /nodejs_test.js
# 웹에서 접근 확인, 콘솔 창에서 ctrl + c를 실행하면 서비스가 종료

# rename을 통한 컨테이너 이름 변경
# nodejs_test에서 nodeapp으로 바뀌었다.
docker rename nodejs_test nodeapp
```



## 볼륨 활용타입

### volume

- 도커에서 권장하는 방법

- 생성 : docker volume create 볼륨이름 
- 도커 볼륨은 도커 명령어를 통해 관리할 수 있음
- 여러 컨테이너 간에 안전하게 공유할 수 있음

```
# 볼륨 생성
docker volume create my-appvol-1

# 볼륨 조회
docker volume ls

# 볼륨 검사, 볼륨이 올바르게 생성되고 마운트되었는지 확인하는 데 사용
docker volume inspect my-appvol-1

# --mount 옵션을 이용한 볼륨 지정
docker run -d --name vol-test1 --mount source=my-appvol-1,target=/app ubuntu:20.04

# -v 옵션을 이용한 볼륨 지정
# 사전에 docker volume create를 하지 않아도 호스트 볼륨 이름을 쓰면 자동 생성
docker run -d --name vol-test2 -v=my-appvol-2:/var/log ubuntu:20.04

# 볼륨 제거, 현재 연결된 컨테이너가 있으면 에러 발생
docker volume rm my-appvol-1

# 연결된 컨테이너 제거 후 볼륨 삭제
docker stop vol-test1
docker rm vol-test1
docker volume rm my-appvol-1
```



### bind mount

- 볼륨 기법에 비해 사용이 제한적
- 호스트 파일 시스템 절대경로: 컨테이너 내부 경로를 직접 마운트하여 사용



### 활용

- 데이터 베이스의 데이터 지속성 유지

```
# 볼륨 생성
docker volume create mysql-data-vol

# 볼륨을 포함한 MySQL 컨테이너 실행
# dockertest 생성
docker run -it --name=mysql-vtest -e MYSQL_ROOT_PASSWORD=mhylee -e MYSQL_DATABASE=dockertest -v mysql-data-vol:/var/lib/mysql -d mysql:5.7

# 컨테이너 접속
docker exec -it mysql-vtest bash

/etc/init.d/mysql start
my -uroot -p
mysql> show databases;
mysql> use dockertest;
# 테이블 생성
mysql> create table mytab(c1 int, c2 char);
# 열 넣기
mysql> insert into mytab values (1, 'a');
# 조회
mysql> select * from mytab;
+------+------+
| c1   | c2   |
+------+------+
|    1 | a    |
+------+------+
# 종료
mysql> exit
조회
ls /var/lib/mysql/dockertest/

docker inspect --format="{{ .Mounts }}" mysql-vtest
# 데이터베이스 컨테이너 장애를 가정해서 정지 후 제거한 뒤 동일 볼륨을 지정, 기존 데이터가 그대로 유지됨을 확인할 수 있음.
docker stop mysql-vtest
docker rm mysql-vtest


docker run -it --name=mysql-vtest -e MYSQL_ROOT_PASSWORD=mhylee -e MYSQL_DATABASE=dockertest -v mysql-data-vol:/var/lib/mysql -d mysql:5.7

mysql -uroot -p
mysql> show databases;
mysql> use docker test;
mysql> show tables;
# 조회
# 기존에 작업했었던 DB 내용인 것을 알 수 있음.
mysql> select * from mytab;
+------+------+
| c1   | c2   |
+------+------+
|    1 | a    |
+------+------+
# 종료
mysql> exit
```

- 컨테이너 간 데이터 공유를 위한 데이터 컨테이너 만들기
  - 컨테이너 볼륨으로 지정된 디렉터리로부터 볼륨 마운트를 할 수 있다.
  - 데이터 컨테이너를 만들 수 있고, 컨테이너 내의 데이터베이스 백업, 복구 및 마이그레이션 등의 작업에 활용할 수 있다.

```
# 도커 볼륨을 통해 데이터 전용 컨테이너 생성
# 명시적으로 docker volume create 볼륨명 으로 볼륨을 생성하지 않아도 
# -v 옵션 사용 시 하나의 디렉터리만 지정하게 되면 호스트에는 볼륨 경로에 임의의 이름으로 생성되고,
# --volumes-from으로 지정된 컨테이너에는 모두 동일한 이름의 디렉터리가 생성된다.
docker create -v /data-volume --name=datavol ubuntu:18.04
# 생성한 데이터 전용 컨테이너를 --volumes-from 옵션을 이용해 공유 연결을 할 2개의 컨테이너 실행.
docker run -it --volumes-from datavol ubuntu:18.04

echo 'testing data container' > /data-volume/test-volume.txt
cat /data-volume/test-volume.txt
ls /data-volume/
exit
docker ps -a


docker run -it --volumes-from datavol ubuntu:18.04
echo 'testing data container2' > /data-volume/test-volume2.txt
cat /data-volume/test-volume2.txt
ls /data-volume/
```

- 실무에서 유용한 볼륨 활용
  - Nginx 웹 서비스를 하는 컨테이너를 개발하고 있다고 가정하자
  - Dockerfile을 이용한 초기 이미지 개발 시 개발 팀으로부터 전달받은 웹 소스를 Nginx 컨테이너 내부의 웹 기본 경로인 /var/www/html에 Dockerfile의 copy로 포함할 수 있다.
  - 이렇게 생성된 이미지를 컨테이너로 실행한 뒤 웹 소스 변경이 있다면 수정된 웹 소스를 docker cp 명령을 통해 다시 넣을 수 있다.
  - 이때 컨테이너 실행 시 볼륨을 지정했다면 애써 docker cp 명령을 사용하지 않고도 해당 볼륨 경로에 변경된 웹 소스만 넣어주면 바로 적용이 가능하다.



## Dockerfile(도커 파일)

### Dockerfile 명령어

- FROM

  - 생성하려는 이미지의 베이스 이미지 지정

  - hub.docker.com에서 제공하는 공식 이미지를 권장

  - 이미지를 선택할 때 작은 크기의 이미지(slim)와 리눅스 배포판인 알파인(Alpine) 이미지를 권장한다.

  - 태그를 넣지 않으면 latest로 지정됨.

    ```dockerfile
    FROM ubuntu:18.04
    ```

- MAINTAINER

  - 일반적으로 이미지를 빌드한 작성자 이름과 이메일을 작성

    ```dockerfile
    NAINTAINER hyunwoo.lee <wkfwktka@danawa.com>
    ```

- LABEL

  - 이미지 작성 목적으로 버젼, 타이틀, 설명, 라이선스 정보 등을 작성. 1개 이상 작성 가능

  - 사용법

    ```dockerfile
    LABEL purpose = 'Nginx for webserver'
    LABEL version = '1.0'
    LABEL description = 'web service application using Nginx'
    ```

  - 권장 사항

    ```dockerfile
    LABEL purpose = 'Nginx for webserver' \
    	version = '1.0' \
    	description = 'web service application using Nginx'
    ```

- RUN

  - 설정된 기본 이미지에 패키지 업데이트, 각종 패키지 설치, 명령 실행 등을 작성. 1개 이상 작성 가능

  - 사용 방법

    ```dockerfile
    RUN apt update
    RUN apt -y install nginx
    RUN apt -y install git
    RUN apt -y install vim
    RUN apt -y install curl
    ```

  - 권장 사항

    - 다단계 빌드 사용 권장, 각 이미지별로 개별 Dockerfile로 빌드.
    - RUN 명령어의 개별 명령 수를 최소화하기 위해 여러 설치 명령을 연결하면 이미지의 레이어 수 감소
    - autoremove, autoclean, rm -rf/var/lib/apt/lists/*을 사용하면 저장되어 있는 apt 캐시가 삭제되므로 이미지 크기가 감소

    ```dockerfile
    # shell 방식
    RUN apt update && apt install -y nginx git vim curl &&\
    	apt-get clean -y && \
    	apt-get autoremove -y && \
    	rm -rfv /tmp/* /var/lib/apt/lists/* /var/tmp/*
    # Exec 방식
    RUN ["/bin/bash", "-c", "apt update"]
    RUN ["/bin/bash", "-c", "apt -y install nginx git vim curl"]
    ```

- CMD

  - 생성된 이미지를 컨테이너로 실행할 때 실해오디는 명령

  - ENTRYPOINT 명령문으로 지정된 커맨드에 디폴트로 넘길 파라미터를 지정할 때 사용한다.

  - **여러 개의 CMD를 작성해도 마지막 하나만 처리됨**

  - 일반적으로 이미지의 컨테이너 실행 시 애플리케이션 데몬이 실행되도록 하는 경우 유용

  - 사용방법

    ```dockerfile
    # Shell 방식
    CMD apachectl -D FOREGROUND
    # Exec 방식
    CMD ["/usr/sbin/apachectl", '-D', "FOREGROUND"]
    CMD ["nginx", "-g", "daemon off;"]
    CMD ["python", "app.py"]
    ```

- ENTRYPOINT

  - CMD와 마찬가지로 생성된 이미지가 컨테이너로 실행될 때 사용하지만, 컨테이너가 실행될 때 명령어 및 인자 값을 전달하여 실행한다는 점이 다르다.

  - 여러 개의 CMD를 사용하는 경우 ENTRYPOINT 명령문과 함께 사용

  - ENTRYPOINT는 커맨드를 지정하고, CMD는 기본명령을 지정하면 탄력적으로 이미지를 실행할 수 있다.

  - ex) python명령을 기본으로 runapp.py 코드를 실행한다면,

    ```dockerfile
    ENTRYPOINT ["python"]
    CMD ["runapp.py"]
    ```

  - 사용 예

    - 동일 환경에 entrypoint.sh 셸 스크립트를 이미지에 넣고(ADD) 실행 권한 설정(RUN) 후 컨테이너 실행 시 entrypoint.sh를 실행(ENTRYPOINT).

    ```dockerfile
    ...
    ADD ./entrypoint.sh /entrypoint.sh
    RUN chmod +x /entrypoint.sh
    ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
    ```

  - CMD와 ENTRYPOINT 비교

    - ENTRYPOINT는 도커 컨테이너 실행 시 항상 수행해야 하는 명령어를 지정(ex. 웹서버, DB등의 데몬 실행)
    - CMD는 도커 컨테이너 실행 시 다양한 명령어를 지정하는 경우 유용

- COPY

  - 호스트 환경의 파일, 디렉터리를 이미지 안에 복사하는 경우 작성

  - 단순한 복사 작업만 지원

  - 빌드 작업 디렉터리 외부의 파일은 COPY할 수 없음

  - 사용법

    ```dockerfile
    COPY index.htmnl /usr/share/Nginx/html
    COPY ./runapp.oy
    
    # 주의
    COPY ./runapp.py 작업 영역 전체를 COPY하므로 비효율적임.
    ```

- ADD

  - 호스트 환경의 파일, 디렉터리를 이미지 안에 복사하는 경우뿐만 아니라 URL 주소에서 직접 다운로드하여 이미지에 넣을 수도 있고, 압축 파일(tar, tar.gz)인 경우에는 지정한 경로에 압축을 풀어서 추가한다.

  - 빌드 작업 디렉터리 외부의 파일은 ADD할 수 없고, 디렉터리 추가 시에는 /로 끝나야 한다.

  - 사용 방법

    ```dockerfile
    ADD index.html /usr/share/nginx/html
    ADD http://example.com/view/customer.tar.gz/workspace/data/
    ADD website.tar.gz /var/www/html
    ```

- ENV

  - 이미지 안에 각종 환경 변수를 지정하는 경우 작성

  - 애플리케이션 사용을 쉽게 하려면 사전에 구성되어야 하는 환경 변수들이 있다.

  - 예를 들어, 자바 홈 디렉터리, 특정 실행 파일의 경로를 보장하기 위해 절대 경로 지정을 위한 PATH 설정, 프로그램 버전 등을 사전에 설정한다.

  - 반복된 표현이 사용되는 경우에도 환경 변수 설정을 권장

  - Dockerfile에서 ENV를 설정하면 RUN, WORKDIR 등에서 환경 변수를 사용해 반복을 피할 수 있다.

  - 사용 방법

    ``` dockerfile
    ENV JAVA_NOME / usr/lib/jvm/java-8-oracle
    ENV PATH /usr/local/nginx/bin:$PATH
    ENV Python 3.9
    사용 예
    ```

- EXPOSE

  - 컨테이너가 호스트 네트워크를 통해 들어오는 트래픽을 리스닝하는 포트와 프로토콜을 지정하기 위해 작성

  - Nginx나 apache는 기본 포트로 HTTP 80번과 HTTPS 443번 포트를 사용하고, 컨테이너 모니터링 이미지로 사용하는 Cadvisor 컨테이너는 8080번 포트를 사용한다.

  - 이미지 내에 애플리케이션이 사용하는 포트를 사전에 확인하고 호스트와 연결되도록 구성하는 경우에 설정하고, docker run 사용시 -p 옵션을 통해 사용한다.

  - 사용 방법

    ```dockerfile
    EXPOSE 80 또는 EXPOSE 80/tcp
    EXPOSE 443
    EXPOSE 8080/udp
    ```

- VOLUME

  - 볼륨을 이미지 빌드에 미리 설정하는 경우 작성

  - 도커 컨테이너에서 사용된 파일과 디렉터리는 컨테이너 삭제와 함께 사라진다. 따라서 사용자 데이터의 보존과 지속성을 위해 볼륨 사용을 권장

  - 볼륨으로 지정된 컨테이너의 경로는 볼륨의 기본 경로 /var/lib/docker와 자동으로 연결됨

  - 사용 방법

    ```dockerfile
    VOLUME /var/log
    VOLUME /var/www/html
    VOLUME /etc/nginx
    # HOST OS의 Volume 기본 경로와 container 내부의 /project 연결
    VOLUME ["project"]
    ```

- USER

  - 컨테이너의 기본 사용자는 root다. 애플리케이션이 권한 없이 서비스를 실행할 수 있다면 USER를 통해 다른 사용자로 변경하여 사용한다.

  - 사용 방법

    ```dockerfile
    RUN ['useradd', 'Jenu']
    USER kevinlee
    RUN ["/bin/bash", "-c", "date"]
    또는,
    RUN groupadd -r mongodb && =
    useradd --no-log-init -r -g mongodb mongodb
    ```

- WORKDIR

  - 컨테이너상에서 작업할 경로(디렉터리) 전환을 위해 작성

  - WORKDIR을 설정하면 RUN, CMD, ENTRYPOINT, COPY, ADD 명령문은 해당 디렉터리를 기준으로 실행

  - 지정한 경로가 없으면 자동 생성되고, 컨테이너 실행 이후 컨테이너에 접속(docker exet -it my_container bash)하면 지정한 경로로 연결된다.

  - 사용 방법

    ```dockerfile
    WORKDIR /workspace
    WORKDIR /usr/share/nginx/html
    WORKDIR /go/src/app
    ```

- ARG

  - docker build 시점에서 변숫값을 전달하기 위해 --build-arg=인자=인자를 정의하여 사용

  - 비밀 키, 계정 비밀번호 같은 민감한 정보 사용 시 이미지에 그대로 존재하여 노출될 위험이 있으므로 주의

  - 사용 방법

    ```dockerfile
    --Dockerfile에 ARG 변수를 저장하고,
    ARG db_name
    --docker build 시 변숫값을 저장하면 이미지 내부로 인자가 몰린다.
    $ docker build --build-arg db_name=jpub_db ,
    CMD db_start.sh -h 127.0.0.1 -d ${b_name}
    ```

- ONBUILD

  - 처음 이미지 빌드에 포함하지만 실행되지 않고, 해당 이미지가 다른 이미지의 기본 이미지로 사용되는 경우 실행될 명령을 지정할 때 장성한다.

  - ONBUILD 명령은 부모 Dockerfile이 자식 Dockerfile에 전달하는 방식이다.

  - 예를들어 1차 개발에서 환경을 만들어주고, 2차 개발에서 ONBUILD에 지정된 소스를 실행하는 것이다.

  - 사용 방법

    ```dockerfile
    --1차 Dockerfile 빌드 시 ONBUILD 포함.
    ONBUILD ADD websource.tar.gz /usr/share/nginx/html/
    
    --2차 Dockerfile에 1차에서 생성된 이미지를 지정하면 ONBUILD에 지정된 ADD 명령이 실행되어 새로운 이미지로 생성된다.
    ```

- STOPSIGNAM

  - docker stop 명령은 컨테이너에게 SIGTERM을 보내 정지한다. 이때 다른 시그널을 넣고자 하는 경우 작성한다.

  - 사용 방법

    ```dockerfile
    STOPSIGNAL SIGKILL	# 시그널 번호 또는 이름
    ```

- SHELL

  - Dockerfile 내부에서 사용할 기본 셸을 지정하는 경우 작성

  - 기본값으로 '/bin/sh'가 지정됨

  - 사용 방법

    ```dockerfile
    SHELL ["/bin/bash", "-c"]
    RUN echo "DOcker world!"
    ```

- HEALTHCHECK

  - 컨테이너의 프로세스 상태를 체크하고자 하는 경우에 작성

  - HEALTHCHECK는 하나의 명령만이 유효하고, 여러 개가 지정된 경우 마지막에 선언된 HEALTHCHECK가 적용됨

  - HEALTHCHECK 옵션

    | 옵션            | 설명           | 기본값 |
    | --------------- | -------------- | ------ |
    | --interval=(초) | 헬스 체크 간격 | 30s    |
    | --timeout=(초)  | 타임 아웃      | 30s    |
    | --retries=N     | 타임 아웃 횟수 | 3      |

  - HEALTHCHECK 상태 코드

    | EXIT 코드    | 설명                                   |
    | ------------ | -------------------------------------- |
    | 0: success   | 컨테이너가 정상적이고 사용 가능한 상태 |
    | 1: unhealthy | 컨테이너가 올바르게 작동하지 않는 상태 |
    | 2: starting  | 예약된 코드                            |

  - docker container inspect [컨테이너명] 또는 docker ps에서 확인할 수 있다.

  - 사용 방법

  - 1분마다 CMD에 있는 명령을 실행하여 3초 이상이 소요되면 한 번의 실패로 간주하고 5번의 타임 아웃이 발생하면 컨테이너의 상태를 "unhealthy"로 변경한다.

    ```dockerfile
    HEALTHCHECK --interval=1m --timeout=3s --retries=5 \
    	CMD curl -f http://localhost || exit 1
    ```

ex) ubuntu환경위에서 python 설치

```dockerfile
FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install python -y
```

실습 : 우분투 환경 위에서 python3 설치와 code-server 작동시키기

```dockerfile
FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install python3 -y
RUN apt-get install wget -y
# RUN apt-get install python-pip -y
# RUN apt-get iputils-ping -y
# RUN apt-get install python-mysqldb -y
# RUN apt-get install python-pymysql -y

RUN wget https://github.com/coder/code-server/releases/download/v4.0.1/code-server_4.0.1_amd64.deb
RUN dpkg -i code-server_4.0.1_amd64.deb
CMD [ "code-server", "--bind-addr", "0.0.0.0:10000" ]
```



## Docker-compose(도커 컴포즈)

### 실습

./test의 Dockerfile

```dockerfile
FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install python3 -y
RUN apt-get install wget -y
# RUN apt-get install python-pip -y
# RUN apt-get iputils-ping -y
# RUN apt-get install python-mysqldb -y
# RUN apt-get install python-pymysql -y

RUN wget https://github.com/coder/code-server/releases/download/v4.0.1/code-server_4.0.1_amd64.deb
RUN dpkg -i code-server_4.0.1_amd64.deb
CMD [ "code-server", "--bind-addr", "0.0.0.0:10000" ]
```

.env

```
DB_ROOT_PASSWORD=root
DB_DATABASE=dockertest
DB_USER=lhw
DB_PASSWORD=1234
```

docker-compose.yml

```yml
version: '3.8'
services:
  DB:
    image: mysql:5.7
    environment:
      MYSQL_ROOT_PASSWORD: "${DB_ROOT_PASSWORD}"
      MYSQL_DATABASE: "${DB_DATABASE}"
      MYSQL_USER: "${DB_USER}"
      MYSQL_PASSWORD: "${DB_PASSWORD}"
    command: mysqld --character-set-server=utf8 --collation-server=utf8_general_ci
    ports:
      - "3306:3306"
    
  test:
    build: ./test
    ports:
      - "10000:10000"
```

code-server

```python
# 접속 : localhost:10000
# apt-get install python-pip -y
# apt-get iputils-ping -y
# apt-get install python-mysqldb -y
# apt-get install python-pymysql -y
import pymysql

print('start')
conn = pymysql.connect(host='DB', port=3306, user="lhw", password="1234")
cur = conn.cursor()

print('show')
cur.execute("SHOW DATABASES")
print(cur)
for i in cur:
    print(i)

print('create')
cur.execute("create table dockertest.test1(id int, name varchar(30), age int);")

print('select, insert')
sql = "select * from dockertest.test1"
into_sql = 'insert into dockertest.test1 values("1", "hw", "24")'
# select
cur.execute(sql)
datas = cur.fetchall()
print(datas)
# insert
cur.execute(into_sql)
cur.execute(sql)
datas = cur.fetchall()
print(datas)
# conn.commit()


cur.close()
conn.close()
print('end')

# start
# show
# <pymysql.cursors.Cursor object at 0x7f1bc8e3af50>
# ('information_schema',)
# ('dockertest',)
# create
# select, insert
# ()
# ((1, 'hw', 24),)
# end
```





## Docker Swarm mode(도커 스웜 모드)

> 동일한 컨테이너를 공유하는 여러 클러스터 내의 노드에서 애플리케이션을 원활하게 실행할 수 있도록 하는 도커 자체 컨테이너 오케스트레이션 도구

### 기본 세팅

- Virtualbox가 필요
- 학습 내용은 virtualbox에서 ubuntu를 설치하여 실행할 예정
- 3개의 노드를 생성할 것임

### 주요 기능

- 분리된 분산 설계
  - 매니저 노드
  - 리더 노드
  - 작업자 노드
- 서비스 확장과 원하는 상태 조정

- 서비스 스케줄링
- 로드밸런싱
- 서비스 검색
  - 도커 스웜 모드는 서비스 검색을 위해 자체 DNS 서버를 통한 서비스 검색 기능을 제공
- c롤링 업데이트

### 도커 스웜 모드 클러스터 구성

#### 도커 스웜 모드 구성을 위한 서버 구성

- Oracle VirtualBox 기반의 우분투 이미지를 복제하여 3개로 만든 뒤 스웜 모드 클러스터를 위한 서버 구성을 수행

| 노드         | 운영체제     | CPU    | Memory | IP 주소        |
| ------------ | ------------ | ------ | ------ | -------------- |
| swrm-manager | ubuntu 20.04 | 4 core | 4 GB   | 192.168.56.100 |
| swrm-worker1 | ubuntu 20.04 | 2 core | 4 GB   | 192.168.56.101 |
| swrm-worker2 | ubuntu 20.04 | 2 core | 4 GB   | 192.168.56.102 |

1. 사용 중인 도커 호스트를 중지하고, 복제를 수행한다.
2. 복제 구성에 필요한 내용을 다음과 같이 변경하고 [다음]을 클릭한다.
   1. 머신 이름 변경: swrm-worker1(머신 이름은 스웜 모드 구성에 맞게 설정하는 사용자 지정이다.)
   2. MAC 주소 정책 변경 : [모든 네트워크 어댑터의 새 MAC 주소 생성]을 선택해 새로운 주소를 할당받도록 한다.
3. 복제 방식을 선택한다. 완전한 복제를 선택하여 기존 이미지의 모든 것을 복제한다.

- 이러한 방식으로 원하는 작업자 노드 수만큼 복제를 수행하면 된다.

- 복제  완료 시 각 노드에 접속하여 호스트명과 IP 주소를 수동으로 할당하여 각 노드가 충돌하지 않도록 서버 구성을 해야 한다.

```
# 매니저 노드 생성
# 도커 스웜 모드의 노드가 매니저 노드에 접근하기 위한 IP를 입력
docker swarm init --advertise-addr ---.---.---.---

# 노드가 매니저 노드와 함께 클러스터에 합류할 수 있도록 해줌
# 내용중에 docker swarm join --token ~~~~을 복사
worker:~$ docker swarm join --token ~~~~

# 매니저 노드에서 작업자 노드의 연결을 확인할 수 있음
swarm-manager:~$ docker node ls

# 운영 중 노드의 확장을 위해 새로운 토큰이 필요한 경우
# --rotate 플래그를 사용하여 새 조인 토큰 생성
docker swarm join-token --rotate worker

# 조인 토큰만 새로 발급하는 경우에는 --quiet옵션
docker swarm join-token -q worker

# 노드 제거 명령
# 매니저노드에서 worker노드 제거
manager:~$ docker swarm leave worker

# worker노드에서 제거
worker:~$ docker swarm leave

#매니저 노드에서 서비스 생성
# 이름, 포트설정, 룰설정, 복제 몇개 만들건지
docker service create --name=이름 --publish=숫자:숫자 --constraint node.role=worker --replicas 3 nginx
# 만약 노드2개에 복제3개가 있으면 한 노드에 컨테이너가 두개 생성됨.

# 서비스 조회
docker service ls
# 서비스에서 실행중인 컨테이너 정보 조회/ 로그의 기능도함
docker service ps 서비스이름

# docker swarm update 내용 서비스이름

# 서비스 갯수 조정
docker service scale 서비스이름=갯수

# 롤백
docker service rollback my-database2

# 서비스 삭제
docker service rm 서비스이름
```





## AWS

### EC2

- 클라우드 서비스 내에서 동작하는 인스턴스 그 무언가
- 보안그룹 : 방화벽이라 생각하면 됨
- ssh연결시

```txt
[AWS ssh 포트 변경]
#!/bin/bash -ex
perl -pi -e 's/^#?Port 22$/Port 10000/' /etc/ssh/sshd_config
service sshd restart || service ssh restart

[AWS EC2 인스턴스 접근]
ssh -i "pattern-analysis.pem" -p 10000 ec2-user@ec2-3-36-53-1.ap-northeast-2.compute.amazonaws.com

# 퍼미션이 너무 공개되어서 접속 안될 수 있으니 프라이빗 키 퍼미션 변경
chmod 600 ~/.ssh/your-key.pem

# ~/.ssh 디렉토리에 키가 없을 시
# cp : 복사 명령어
# /mnt/c/Users/admin/.ssh/에 있는 lhw-ojt.pem을
# ~/.ssh에 복사하겠다는 의미
cp /mnt/c/Users/admin/.ssh/lhw-ojt.pem ~/.ssh
```

- 도커 설치

```
- yum업뎃
sudo yum update -y
- 도커설치
sudo amazon-linux-extras install docker
-도커 시작
sudo service docker start
- 권한 : ec2-user를 사용하지 않고도 도커 명령을 실행할 수 있도록 docker 그룹에 sudo를 추가합니다.
sudo usermod -a -G docker ec2-user
# 안되면 밑의 명령어
sudo setfacl -m user:ec2-user:rw /var/run/docker.sock

docker pull wkfwktka/project:0.0.1

docker run -p 5000:5000 -it --name project1 wkfwktka/project:0.0.1
```





### S3

- 버킷
  - 한마디로 폴더로 봐도 무방하다
  - 데이터를 저장하는 폴더
  - C드라이브나 D드라이브 같은 아주 큰~ 저장소로 보면됨
  - 클라우드상에서 스토리지
  - 다운로드나 업로드가 가능한데 권한이 필요함
  - 권한은 키(IAM)를 사용함.

### IAM

- 자격증명을 해주는 키라고 보면 될 듯
- 사용자로 만들 수도 있지만 그룹화도 가능

### VPC

- 가상 버츄얼 프라이빗
- DHCP 옵션 세트
  - ip 재할당해주는 옵션

### Route53

- 아마존에서 관리하는 DNS(도메인 네임 서버)에 내 주소를 등록하는 것

### CloudWatch

- 모니터링 서비스

### Lambda

- 유저 -> URL호출(트리거) -> API 게이트웨이 -> lambda를 실행 -> 결과 -> 유저

### Elastic Cache

- save같은 녀석임
- 노트북검색 -> 방대한 양을 계속 시간들여 주는것은 낭비
- 전에 검색해봤으면 그 내용을 갖고있다가 바로보여주는것





```
sudo /etc/init.d/docker start
```

