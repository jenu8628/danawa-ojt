import pandas as pd
import numpy as np
from flask import Flask, render_template
from flask import request
import pickle

def predict_data(model, test_input):
    # 예측값 반환 함수
    # model : 학습된 모델
    # test_input : 테스트 입력 데이터
    return model.predict(test_input)

app = Flask(__name__)

@app.route(rule="/")
def home():
    return render_template("main.html")


@app.route(rule="/wine/", methods=['GET', 'POST'])
def wine():
    if request.method == 'POST':
        chlorides = float(request.form.get('chlorides'))
        volatile_acidity = float(request.form.get('volatile_acidity'))
        density = float(request.form.get('density'))
        alcohol = float(request.form.get('alcohol'))
        wine_type = int(request.form.get('wine_type'))
        test_input = {
            'chlorides' : [chlorides],
            'volatile acidity' : [volatile_acidity],
            'density' : [density],
            'alcohol' : [alcohol],
            'wine_type' : [wine_type]
        }
        df = pd.DataFrame(test_input, columns=['chlorides', 'volatile acidity', 'density', 'alcohol', 'type'])
        with open('winePickle.pickle', 'rb') as f:
            model= pickle.load(f)
        result = predict_data(model, df)[0]
        return render_template('wine.html', 
        chlorides=chlorides, 
        volatile_acidity=volatile_acidity, 
        density=density, 
        alcohol=alcohol,
        wine_type=wine_type, 
        result=result,
        site="POST"
        )
    else:
        return render_template('wine.html', site='GET')


@app.route(rule="/titanic/", methods=['GET', 'POST'])
def titanic():
    if request.method == 'POST':
        sex = request.form.get('sex')
        sex = 0 if sex == 'femail' else 1
        Pclass = request.form.get('Pclass')
        if Pclass == 'C':
            Pclass = 0
        elif Pclass == 'Q':
            Pclass = 1
        else:
            Pclass = 2

        age = int(request.form.get('age'))
        family = int(request.form.get('family'))
        if family == 0:
            family = 0
        elif 0 < family <= 3:
            family = 1
        else:
            family = 2
        test_input = {
            'Sex' : [sex],
            'Pclass' : [Pclass],
            'Age' : [age],
            'Family_Size' : [family],
        }
        df = pd.DataFrame(test_input, columns=['Sex', 'Pclass', 'Age', 'Family_Size'])
        with open('titanicPickle.pickle', 'rb') as f:
            model= pickle.load(f)
        result = predict_data(model, df)[0]
        result = 'die' if result == 0 else 'survived'

        return render_template('titanic.html', 
        sex=request.form.get('sex'),
        Pclass=request.form.get('Pclass'), 
        age=age,
        family=int(request.form.get('family')),
        result=result,
        site="POST"
        )
    else:
        return render_template('titanic.html', site="GET")


@app.route(rule="/movie/", methods=['GET', 'POST'])
def movie():
    if request.method == 'POST':
        # num_rank : 영화 배급사
        # time : 상영 시간
        # num_staff : 스태프 수
        # num_actor : 주연배우 수
        # genre_rank : 장르
        # screening_rat_12세 관람가
        # screening_rat_15세 관람가  
        # screening_rat_전체 관람가 
        # screening_rat_청소년 관람불가
        with open('movies.pickle', 'rb') as f:
            model = pickle.load(f)
            # 배급사
            dist = pickle.load(f)
            genre_dict = pickle.load(f)
        if request.form.get('distributor') in dist:
            num_rank = np.where(dist == request.form.get('distributor'))[0][0] + 1
        else:
            num_rank = 0
        time = int(request.form.get('time'))
        num_staff = int(request.form.get('num_staff'))
        num_actor = np.log1p(int(request.form.get('num_actor')))
        genre_rank = genre_dict[(request.form.get('genre'))]
        arr = [0] * 4
        arr[int(request.form.get('screening_rat'))] = 1
        test_input = {
            'num_rank' : [num_rank],
            'time' : [time],
            'num_staff' : [num_staff],
            'num_actor' : [num_actor],
            'genre_rank' : [genre_rank],
            'screening_rat_12세 관람가' : [arr[0]],
            'screening_rat_15세 관람가' : [arr[1]],
            'screening_rat_전체 관람가' : [arr[2]],
            'screening_rat_청소년 관람불가' : [arr[3]]
        }
        # 테스트 첫번째
        # 1,342,754
        # 1,342,754 

        df = pd.DataFrame(test_input, columns=['num_rank', 'time', 'num_staff', 'num_actor', 'genre_rank',
            'screening_rat_12세 관람가', 'screening_rat_15세 관람가', 'screening_rat_전체 관람가', 'screening_rat_청소년 관람불가'
        ])
        mid = str(int(np.expm1(predict_data(model, df)[0]).round()))[::-1]
        result = ''
        for i in range(len(mid)):
            if i % 3 == 0 and i != 0:
                result += ','
            result += mid[i]
        result = result[::-1]
        return render_template('movie.html', site="POST", result=result)

    else:
        with open('movies.pickle', 'rb') as f:
            model = pickle.load(f)
            # 배급사
            dist = pickle.load(f)
            genre_dict = pickle.load(f)
        return render_template('movie.html', site="GET", genre=genre_dict, dist=dist)


@app.route(rule='/fifa/', methods=['GET', 'POST'])
def fifa():
    if request.method == 'GET':
        return render_template('fifa.html', site="GET") 

    else:
        # age : 나이
        # continent : 선수들의 국적이 포함되어 있는 대륙 europe, south america, asia, africa, oceania
        # contract_until : 선수의 계약기간이 언제까지 인지 년월일 = 년.월일
        # position : 선수가 선호하는 포지션 MF DF ST GK
        # reputation : 선수가 유명한 정도 1~5
        # stat_overall : 선수의 현재 능력치 0~100
        # stat_potential : 선수가 경험 및 노력을 통해 발전할 수 있는 정도 0~100
        # stat_skill_moves : 선수의 개인기 능력치 1~5
        age = np.log1p(int(request.form.get('age')))
        continent = int(request.form.get('continent'))
        year, month, day = request.form.get('contract_until').split('-')
        contract_until = int(year) + float('0.' + month)
        position = int(request.form.get('position'))
        reputation = int(request.form.get('reputation'))
        stat_overall = int(request.form.get('stat_overall'))
        stat_potential = np.log1p(int(request.form.get('stat_potential')))
        stat_skill_moves = int(request.form.get('stat_skill_moves'))
        continent_arr = [0] * 5
        position_arr = [0] * 4
        continent_arr[continent] = 1
        position_arr[position] = 1
        test_input = {
            'age' : [age],
            'contract_until' : [contract_until],
            'reputation' : [reputation],
            'stat_overall' : [stat_overall],
            'stat_potential' : [stat_potential],
            'stat_skill_moves' : [stat_skill_moves],
            'continent_africa' : [continent_arr[0]],
            'continent_asia' : [continent_arr[1]],
            'continent_europe' : [continent_arr[2]],
            'continent_oceania' : [continent_arr[3]],
            'continent_south america' : [continent_arr[4]],
            'position_DF' : [position_arr[0]],
            'position_GK' : [position_arr[1]],
            'position_MF' : [position_arr[2]],
            'position_ST' : [position_arr[3]]
        }

        # 트레인 첫번째 값
        # 110,500,000
        # 105,547,932

        # 테스트 첫번째 값
        # 6.443993e+07
        # 64,439,930
        # 66,730,106 
        df = pd.DataFrame(test_input, columns=['age', 'contract_until', 'reputation',
            'stat_overall', 'stat_potential', 'stat_skill_moves',
            'continent_africa', 'continent_asia', 'continent_europe', 'continent_oceania', 'continent_south america',
            'position_DF', 'position_GK', 'position_MF', 'position_ST'
        ])
        with open('fifa.pickle', 'rb') as f:
            model = pickle.load(f)
        mid = str(int(np.expm1(predict_data(model, df)[0]).round()))[::-1]
        result = ''
        for i in range(len(mid)):
            if i % 3 == 0 and i != 0:
                result += ','
            result += mid[i]
        result = result[::-1]
        return render_template('fifa.html', site="POST", result=result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
