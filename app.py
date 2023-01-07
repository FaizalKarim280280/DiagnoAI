from flask import Flask, render_template, request
from pred_utils import Model, Predict
import pandas as pd

model = Model().model()
tokenizer = Model().tokenizer()
pred = Predict(model=model, tokenizer=tokenizer)

# remedies csv
df = pd.read_csv("static/src/remedies.csv")

app = Flask(__name__)

# home route
@app.route("/")
def index():      
      return render_template("index.html", prediction_text=[])

@app.route("/")
def sim():
      return render_template("index.html/#Simulation")

# about route
@app.route("/about")
def about():
      return render_template("about.html")

# method route
@app.route("/method")
def method():
      return render_template("methodology.html")

# developers route
@app.route("/developers")
def devs():
      return render_template("developers.html")

# predict route
@app.route("/predict", methods=["post"])
def predict():
      prediction = []
      remedies = []
      prompt = [str(x) for x in request.form.values()][0]          # extract the input description
      raw_prediction = pred.predict_disease(prompt)[0]             # run the predictions
      for i in range(0, 3):                                        # take top 3 classes 
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