from flask import Flask, render_template

app = Flask(__name__, template_folder='./templates', static_folder='../static')


@app.route('/')
def home():
    return render_template('index.html', image_filename='img/happy.webp', display_mode='none')


@app.route('/predict', methods=['POST'])
def predict():
    return render_template('index.html', image_filename='img/happy.webp', display_mode='none')


@app.route('/save_pred', methods=['POST'])
def save_pred():
    return render_template('index.html', image_filename='img/happy.webp', display_mode='none')


if __name__ == '__main__':
    app.run(debug=True)
