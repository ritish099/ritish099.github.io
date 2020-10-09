from flask import Flask, request, render_template
from Fake_News_NB_module import *

app = Flask(__name__)

@app.route('/')
def index():
    text = request.args.get('text')
    res = predictNB(text)
     
    return res
 
if __name__=='__main__':
    app.run(debug=True)
        