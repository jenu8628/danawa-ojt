### Init Repo

- 저장소 생성시 기본 master브랜치만 존재하기 때문에 init repo를 통해 git flow에 맞는 브랜치를 구성

<img src="Git Flow.assets/image-20220107132117251.png" alt="image-20220107132117251"  />

- 완료 되고 나면 develop브랜치로 바뀐다.

![image-20220107132234170](Git Flow.assets/image-20220107132234170.png)

- 현재는 로컬에서만 gitflow가 적용되었으므로 git push를 통해 기본 브랜치를 깃랩에 적용

![image-20220107132348339](Git Flow.assets/image-20220107132348339.png)

![image-20220107132426489](Git Flow.assets/image-20220107132426489.png)

![image-20220107132500044](Git Flow.assets/image-20220107132500044.png)



### 기능 개발은 feature

- Start Feature를 클릭하여 새로운 브랜치를 생성합니다.

![image-20220107132552341](Git Flow.assets/image-20220107132552341.png)

- 기능의 이름과 베이스가 될 브랜치를 선택하고 ok를 눌러줍니다.

![image-20220107132647118](Git Flow.assets/image-20220107132647118.png)

- Git -> Show Git Log를 선택하면 현재 로컬의 브랜치를 알 수 있습니다.
- 여기까지 진행하면 로컬에선 feature가 있지만, 깃랩에는 feature/print-hello가 없는 상태이므로 git push를 하여 깃랩 저장소에도 브랜치가 등록되도록 합니다.

![image-20220107132909355](Git Flow.assets/image-20220107132909355.png)



- 코드 추가

![image-20220107132958883](Git Flow.assets/image-20220107132958883.png)

- 커밋

![image-20220107133016506](Git Flow.assets/image-20220107133016506.png)



![image-20220107133050023](Git Flow.assets/image-20220107133050023.png)

- Push tags 체크

![image-20220107133109168](Git Flow.assets/image-20220107133109168.png)

### merge의 두 가지 방법

> 깃랩을 이용해 create merge request와 merge를 진행하고나면 파이참에서 Finish Feature를 눌렀을 때 오류가 나는 것 같다. 어떻게 해야하는지 물어보자!

1. 깃랩이용

![image-20220107133219534](Git Flow.assets/image-20220107133219534.png)

- merge하려는 브랜치를 잘 확인합시다.
- develop을 merge할 시 Delete체크박스를 해제합니다.
  - 병합 후 브랜치를 삭제하겠다는 내용입니다.

![image-20220107134602841](Git Flow.assets/image-20220107134602841.png)

![image-20220107133425586](Git Flow.assets/image-20220107133425586.png)

2. Finish Feature 이용(깃랩에서 merge만들어서 하는게 충돌 확인 때매 좋을것 같다고 생각들었습니다.)

- Finish Feature를 누르면 로컬에서 자동으로 develop브랜치에 merge됩니다.
- feature/action 브랜치는 삭제됩니다.

![image-20220107134100471](Git Flow.assets/image-20220107134100471.png)

- develop에 merge되었으니 push를 통해 git lab에 반영합니다.

![image-20220107134148548](Git Flow.assets/image-20220107134148548.png)

![image-20220107134134530](Git Flow.assets/image-20220107134134530.png)