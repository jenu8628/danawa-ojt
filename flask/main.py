import pandas as pd
from flask import Flask, render_template
from flask import request
import DataAnalysis

app = Flask(__name__)

@app.route(rule="/")
def home():
    # Pclass = request.args.get('Pclass')
    # Name = request.args.get('Name')
    # Sex = request.args.get('Sex')
    return render_template("main.html")

@app.route(rule="/wine", methods=['GET', 'POST'])
def wine():
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
    df = pd.DataFrame(test_input, columns=['chlorides', 'citric acid', 'density', 'alcohol', 'type'])
    model = DataAnalysis.model
    result = DataAnalysis.predict_data(model, df)[0]
    return render_template('wine.html', 
    username=request.form, 
    chlorides=chlorides, 
    volatile_acidity=volatile_acidity, 
    density=density, 
    alcohol=alcohol,
    wine_type=wine_type, 
    result=result
    )



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
