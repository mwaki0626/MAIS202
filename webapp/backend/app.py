from flask import Flask, render_template, request
from image_predictor import ImageDetectModel

app = Flask(__name__)
trash_detect = ImageDetectModel()


# Base endpoint to perform prediction.
@app.route('/', methods=['POST'])
def make_prediction():
    if request.form['predictor'] == 'trash_detect':
        prediction = trash_detect.pred(request)
        print ("~~~~~")
        print (prediction)
        return render_template('index.html', prediction=prediction, generated_text=None, tab_to_show='trash_detect')


@app.route('/', methods=['GET'])
def load():
    return render_template('index.html', prediction=None, generated_text=None, tab_to_show='trash_detect')


@app.route('/predict/image', methods=['POST'])
def make_image_prediction():
    prediction = trash_detect.pred(request)
    print(prediction)
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)


