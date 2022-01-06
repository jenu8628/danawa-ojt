# matplolib 정리

## 기본 설정

### 한글  폰트 깨짐 방지

```python
matplotlib.rc('font', family = '폰트이름')
```

### 음수(-)값 깨짐 방지

```python
matplotlib.rcParms['axes.unicode_minus'] = False
```

### Style, Color 참조

| Parameter | 내용                |
| --------- | ------------------- |
| -         | 실선                |
| --        | dashed              |
| :         | 점선                |
| .         | 원형마커            |
| D         | 다이아몬드(대)마커  |
| d         | 다이아몬드(소) 마커 |
| ^         | 세모(상)마커        |
| >         | 세모(우)마커        |
| <         | 세모(좌)마커        |
| alpha     | 투명도 옵션         |
| **Color** |                     |
| 'b'       | blue                |
| 'g'       | green               |
| 'r'       | red                 |
| 'y'       | yellow              |
| 'k'       | black               |
| 'w'       | white               |

### 그래프 설정

| Parameter                             | 내용                       |
| ------------------------------------- | -------------------------- |
| plt.figure(figsize=(n,m))             | 그래프 크기 조절           |
| plt.title('text', fontsize=15)        | 그래프 제목 설정           |
| plt.xlim([min, max])                  | x축 범위 설정              |
| plt.ylim([min, max])                  | y축 범위 설정              |
| plt.text(x,y, "text")                 | 텍스트 표시                |
| plt.xlabel('text', fontsize=15)       | x축 라벨 설정              |
| plt.ylabel('text', fontsize=15)       | y축 라벨 설정              |
| plt.grid()                            | 격자 생성                  |
| plt.legend()                          | 범례표시                   |
| plt.savefig('저장경로')               | 그래프 저장                |
| plt.xaxis(rotation=90, fontsize='15') | x축 폰트 회전 및 크기 설정 |
| plt.yaxis(rotation=90, fontsize='15') | y축 폰트 회전 및 크기 설정 |
| plt.style.use('ggplot')               | 그래프 스타일 지정         |



## 꺾은선 그래프 plot()

- plt.plot(x축, y축)
- plt.plot(list)
  - 만약 하나의 리스트가 들어온다면 y축을 나타내게 된다.
  - x축은 0부터 순차적으로 증가
- ''<색깔><마커모양><선모양>"으로 사용가능



## 히스토그램 hist()

- 빈도수 비교 그래프
- plt.hist(list, bins=가로축구간개수)
  - list는 x축 인자가 되며, 리스트의 해당요소 빈도수 만큼 y축에 막대가 올라감
  - bins 옵션으로 몇개의 구간으로 나눌지 결정 가능



## 막대그래프 bar(), barh(), 항아리 모양 그래프

- 특정 범주형 변수(x)와 그 크기(y)를 표현할 때 주로 활용
- bar(막대를 표시할 위치(x)(보통 리스트의 길이가 옴), 막대의 높이(y))
- bar(list)
  - x를 지정하지 않으면 x축은 0부터 시작
  - list의 순서대로 막대그래프 생성

- 수형방향 막대그래프 barh()
- barh(y축 범위, 막대의 높이(x))
  - y축범위 생략 불가
- barh를 이용하여 0을 기준으로 음수 양수를 표현해 항아리 모양 그래프를 그릴 수 있다.



## 상자 그림 boxplot()

- 최솟값, 1사분위, 2사분위, 3사분위, 최대값을 보여주는 그래프
- box(list)
  - list에 맞춰 그래프를 그려준다. 이상치도 보여줌.
- 이상치 값 생략 : showfliers=False



## 산점도(+ 버블 차트)





## 파이 차트 pie()

> 기본적으로 반시계방향 시작위치 90도임

- pie(size, labels, colors, autopct, explode, startangle)
  - size: list를 넣으면 해당요소를 전체로 나눈 비율 만큼의 부채꼴을 만들어줌
  - labels : 리스트 형식으로 넣으면 해당 요소에 이름 써줌
  - colors : 리스트 형식으로 넣으면 해당 요소마다 색깔 지정
  - autopct: auto percent를 의미, 해당 비율만큼의 숫자를 적을 수 있다.
    - '%.1f%%': 소수 1번째자리 까지 표기하겠단 의미로 두번째자리에서 반올림
  - explode : 돌출효과 표시 옵션
    - 4개의 요소가 있는데 (0, 0, 0.1, 0)을 넣으면 3번째 요소가 돌출됨.
  - startangle : 시작위치 정하는 옵션 기본적으로 90도에서 시작하고 숫자를 넣으면 start + 숫자 만큼 반시계방향 부터 시작