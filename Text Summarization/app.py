from flask import Flask, render_template, request, redirect, url_for
from generateSummary import generateSummary

app = Flask(__name__)
summaryObject = generateSummary()


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', flag=False)


@app.route('/generatesummary', methods=['POST'])
def generatesummary():
    article = request.form['article']
    try:
        lines = int(request.form['linesinsummary'])
    except:
        lines = 4
    print(lines)
    # print(article)
    summary = summaryObject.main(article, num_of_lines_in_summary=lines)
    # print(summary)
    return render_template('index.html', flag=True, summary=summary)


app.run(debug=True)
