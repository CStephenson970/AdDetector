from flask import render_template
from flask import request
from AdDetector import app
from lstm_model import lstm_model
import cPickle as pickle

from bs4 import BeautifulSoup
import requests

admodel = lstm_model()
test_text = "This is some sample text to see if things are working properly"
print(admodel.evaluate_text(test_text))

def get_article(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(url, headers=headers)
    raw_html = response.content
    
    soup = BeautifulSoup(raw_html,"lxml")
    lines = soup.find_all('p')
    
    hr_lines = []
    for line in lines:
        text = line.get_text()
        hr_lines.append(text)
    
    article = ' '.join(hr_lines)
    return article    


@app.route('/')
@app.route('/index')
def index():
   tmp_variable = 1
   page_html = render_template("input_page.html")
   return page_html

@app.route('/output')
def get_output():
    input_url = request.args.get('url_input')
    input_text = request.args.get('text_input')

    if len(input_url) > 0:
        article_text = get_article(input_url)
    else:
        article_text = input_text
        
    result = admodel.evaluate_text(article_text)
    
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
       
    best,worst = admodel.get_best_worst(article_text)
    
    page_html = render_template("output_page.html",result_str=result_str,
                                qualifier=qualifier,
                                most_ad_like=worst,
                                least_ad_like=best)
    return page_html