from flask import Flask, render_template, jsonify, request
from questionanswer import QuestionAnswerEngine
from symptomprediction import SymptomPrediction
app = Flask(__name__)
global engine1
global engine2
engine1 = QuestionAnswerEngine()
engine2 = SymptomPrediction()


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())


# @app.route('/chatbot', methods=["GET", "POST"])
# def chatbotResponse():
#     if request.method == 'POST':
#         the_question = request.form['question']
#     print('App ', the_question)
#     response = engine1.user_question_process(the_question)
#     return jsonify({"response": response})

@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST':
        the_question = request.form['question']
    print('App ', the_question)
    response = engine1.universal_sentence_encoder(the_question)
    return jsonify({"response": response})


@app.route('/predictionform', methods=["GET", "POST"])
def predictionForm():
    return render_template('prediction_form.html', **locals())


@app.route('/predictsymptoms', methods=["GET", "POST"])
def predictSymptoms():
    if request.method == 'POST':
        cough = int(request.form['cough'])
        fever = int(request.form['fever'])
        sore_throat = int(request.form['sore_throat'])
        shortness_of_breath = int(request.form['shortness_of_breath'])
        head_ache = int(request.form['head_ache'])
        age_60_and_above = int(request.form['age_60_and_above'])
        gender = int(request.form['gender'])
        travel_histroy = int(request.form['test_indication'])
        contact = int(request.form['contact'])
        if travel_histroy == 1 and contact == 1:
            test_indication = 1
        elif travel_histroy == 1 and contact == 0:
            test_indication = 1
        elif travel_histroy == 0 and contact == 1:
            test_indication = 1
        elif travel_histroy == 0 and contact == 0:
            test_indication = 0
        symptom_list = [cough, fever, sore_throat, shortness_of_breath,
                        head_ache, age_60_and_above, gender, test_indication]
    print('Symptom List ', symptom_list)
    response = engine2.predictSymptom(symptom_list)
    return render_template('prediction_form.html', flag=True, response=response)


if __name__ == '__main__':
    app.run(debug=True)
