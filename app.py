from flask import Flask, render_template, request
from pred_utils import Model, Predict
import pandas as pd

model = Model().model()
tokenizer = Model().tokenizer()
pred = Predict(model=model, tokenizer=tokenizer)
df = pd.read_csv("static/src/remedies.csv")

app = Flask(__name__)

@app.route("/")
def index():      
      return render_template("index.html", prediction_text=[])

@app.route("/")
def sim():
      return render_template("index.html/#Simulation")

@app.route("/about")
def about():
      return render_template("about.html")

@app.route("/method")
def method():
      return render_template("methodology.html")

@app.route("/developers")
def devs():
      return render_template("developers.html")

@app.route("/predict", methods=["post"])
def predict():
      prediction = []
      remedies = []
      prompt = [str(x) for x in request.form.values()][0]
      raw_prediction = pred.predict_disease(prompt)[0]
      for i in range(0, 3):
            prediction.append(raw_prediction[i])
      for i in range(0,3):
            x = df[df['Disease'] == prediction[i]['label']]
            y = x.to_dict('list')
            remedies.append(y['Score'][0])
      return render_template("index.html", prediction_text=prediction, remedies_list = remedies)

def main():
      print("Server has started")
      app.run(debug=True, port=3000)

if __name__ == "__main__":
      main()