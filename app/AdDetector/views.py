from flask import render_template
from flask import request
from AdDetector import app
from lstm_model import lstm_model
import cPickle as pickle

import requests
import justext

admodel = lstm_model()
test_text = "This is some sample text to see if things are working properly"
print(admodel.evaluate_text(test_text))

def get_article(url):
    #url = "https://boilerpipe-web.appspot.com/extract?url=" + url + "&extractor=ArticleExtractor&output=htmlFragment&extractImages=&token="
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(url, headers=headers)
    raw_html = response.content
    
    article = ''
    paragraphs = justext.justext(response.content, justext.get_stoplist("English"))
    
    for paragraph in paragraphs:
        if not paragraph.is_boilerplate:
            article += ' ' + paragraph.text
    
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
    
    if len(article_text) <= 100:
        page_html = render_template("null_output.html")
        return page_html
        
    result = admodel.evaluate_text(article_text)
    
    is_ad = False
    if result[0] <= 0.5:
       result_str = "This looks like sponsored content!"
       is_ad = True
    if result[0] > 0.5:
       result_str = "This looks like a regular article!"
       is_ad = False
    
    if is_ad == True:
       confidence = round(result[1],3)*100
    if is_ad == False:
       confidence = round(result[0],3)*100
       
    best,worst = admodel.get_best_worst(article_text)
    
    page_html = render_template("output_page.html",result_str=result_str,
                                most_ad_like=worst,
                                processed_text=article_text)
    return page_html

@app.route('/checker', methods=['POST'])
def check_link():
    url = request.form['url']
    try:
        article_text = get_article(url)
    except:
        return render_template("badcheck.html")
    
    if len(article_text) == 0:
        return render_template("badcheck.html")
            
    result = admodel.evaluate_text(article_text)
    ad_chance = round(result[1],3)*100
    if ad_chance >= 50:
        response_str = "This looks like sponsored content!"
    if ad_chance < 50:
        response_str = "This looks like a regular article!"
        
    page_html = render_template("checkpage.html",response_str=response_str)
    return page_html