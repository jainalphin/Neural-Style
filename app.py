from flask import Flask, render_template,  request
from prediction import  *

app = Flask(__name__)

@app.route('/', methods = ['POST','GET'])
def index():
    return render_template ('index.html') #This line will render files from the folder templates

@app.route('/predict',methods = ['POST', 'GET'])
def predict():
    content = request.files['file']
    content_path = 'static/images/uploads/content.jpg'
    content.save(content_path)

    style = request.form.get('style')
    if style == None:
        style = request.files['file1']
        style_path = 'static/images/uploads/style.jpg'
        style.save(style_path)
    else:
        style_path = 'static/style_image/' + str(style) + '.jpg'

    image_path = pred(content_path,style_path)
    return render_template('result.html',image_path = image_path)



if __name__ == '__main__':
    app.run(debug=True)
