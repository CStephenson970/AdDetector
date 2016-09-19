from flask import render_template
from flask import request
from AdDetector import app
from lstm_model import lstm_model
import cPickle as pickle

admodel = lstm_model()
text = "this is some sample text to see if things are working properly"
print(admodel.evaluate_text(text))

@app.route('/')
@app.route('/index')
def index():
   tmp_variable = 1
   page_html = render_template("input_page.html")
   return page_html

@app.route('/output')
def get_output():
    input_text = request.args.get('text_input')
    print(input_text)
    result = admodel.evaluate_text(input_text)
    
    is_ad = False
    if result[0] < 0.5:
       result_str = "it's advertising!"
       is_ad = True
    else:
       result_str = "it's not advertising!"
       is_ad = False
    
    if is_ad == True:
       confidence = round(result[1],3)*100
    if is_ad == False:
       confidence = round(result[0],3)*100
    
    if  0.3 < result[0] < 0.7:
       qualifier = "But I'm only " + str(confidence) + "%" + " sure about it."
    else:
       qualifier = "I'm " + str(confidence) + "%" + " sure about it."
       
    page_html = render_template("output_page.html",result_str=result_str,qualifier=qualifier)
    return page_html