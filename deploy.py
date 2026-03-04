from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# load the model
GBC = pickle.load(open('GradientBoostingClassifier()model.pkl', 'rb'))
ADA = pickle.load(open('AdaBoostClassifier()model.pkl', 'rb'))
DTC = pickle.load(open('DecisionTreeClassifier()model.pkl', 'rb'))
GNB = pickle.load(open('GaussianNB()model.pkl', 'rb'))
KNC = pickle.load(open('KNeighborsClassifier()model.pkl', 'rb'))
LDA = pickle.load(open('LinearDiscriminantAnalysis()model.pkl', 'rb'))
LRM = pickle.load(open('LogisticRegression()model.pkl', 'rb'))
MLP = pickle.load(open('MLPClassifier()model.pkl', 'rb'))
RFC = pickle.load(open('RandomForestClassifier()model.pkl', 'rb'))
SVC = pickle.load(open('SVC()model.pkl', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    GBC_result = GBC.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    ADA_result = ADA.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    DTC_result = DTC.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    GNB_result = GNB.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    KNC_result = KNC.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    LDA_result = LDA.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    LRM_result = LRM.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    MLP_result = MLP.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    RFC_result = RFC.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    SVC_result = SVC.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)