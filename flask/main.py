import pandas as pd
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
    elif request.method == 'GET':
        return render_template('wine.html', site='GET')


@app.route(rule="/titanic", methods=['GET', 'POST'])
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
    elif request.method == 'GET':
        return render_template('titanic.html', site="GET")


@app.route(rule="/movie", methods=['GET', 'POST'])
def movie():
    return


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
