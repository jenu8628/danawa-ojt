

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

| 옵션  | 설명                                                   |
| :---- | :----------------------------------------------------- |
| -d    | detached mode 흔히 말하는 백그라운드 모드              |
| -p    | 호스트와 컨테이너의 포트를 연결 (포워딩)               |
| -v    | 호스트와 컨테이너의 디렉토리를 연결 (마운트)           |
| -e    | 컨테이너 내에서 사용할 환경변수 설정                   |
| -name | 컨테이너 이름 설정                                     |
| -rm   | 프로세스 종료시 컨테이너 자동 제거                     |
| -it   | -i와 -t를 동시에 사용한 것으로 터미널 입력을 위한 옵션 |
| -link | 컨테이너 연결 [컨테이너명:별칭]                        |



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

```
```





















