import os
import pickle

import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "XGBOOST.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Column order the model was trained with
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]

# Label encodings (alphabetical, as sklearn LabelEncoder produces).
# If you encoded differently in your notebook, change the numbers here.
SEX = [(0, "Female"), (1, "Male")]
SMOKER = [(0, "No"), (1, "Yes")]
REGION = [(0, "Northeast"), (1, "Northwest"), (2, "Southeast"), (3, "Southwest")]

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Insurance Cost Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#15122e; --card:#ffffff; --ink:#1d1a3a; --muted:#6b6890; --line:#e1def2;
    --violet:#6d5dfc; --violet-d:#5446e0; --amber:#ffb547; --amber-d:#f59e0b; --soft:#f6f5fe;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:Inter,system-ui,sans-serif;color:var(--ink);min-height:100vh;padding:32px 16px;
    background:radial-gradient(1000px 600px at 0% -10%,#3b2f8f 0%,transparent 60%),
               radial-gradient(800px 500px at 110% 0%,#7a3b8f 0%,transparent 55%),var(--bg)}
  .wrap{max-width:820px;margin:0 auto}
  header{text-align:center;color:#fff;margin-bottom:24px}
  .badge{display:inline-flex;width:56px;height:56px;border-radius:16px;align-items:center;justify-content:center;
    font-size:28px;background:linear-gradient(135deg,var(--amber),var(--amber-d));margin-bottom:12px}
  h1{font-family:'DM Serif Display',serif;font-weight:400;font-size:2.3rem;margin:0 0 6px}
  header p{margin:0;color:#bdb9e6}
  .card{background:var(--card);border-radius:18px;padding:28px;box-shadow:0 20px 50px rgba(0,0,0,.4)}
  .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px 22px}
  @media(max-width:640px){.grid{grid-template-columns:1fr}.card{padding:20px}}
  label{display:block;font-size:.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
  input,select{width:100%;padding:12px 14px;border:1.5px solid var(--line);border-radius:10px;font:inherit;color:var(--ink);background:var(--soft);transition:.15s}
  input:focus,select:focus{outline:none;border-color:var(--violet);background:#fff;box-shadow:0 0 0 4px rgba(109,93,252,.15)}
  button{margin-top:26px;width:100%;padding:15px;border:0;border-radius:12px;font:600 1.05rem Inter,sans-serif;color:#fff;
    background:linear-gradient(135deg,var(--violet),var(--violet-d));cursor:pointer;transition:.15s}
  button:hover{transform:translateY(-1px);box-shadow:0 10px 22px rgba(84,70,224,.4)}
  .result{margin-top:24px;padding:22px;border-radius:14px;text-align:center;background:linear-gradient(135deg,#fff7e6,#ffecc7);border:1.5px solid #ffd58a}
  .result small{display:block;color:#8a6417;font-weight:600;text-transform:uppercase;letter-spacing:.06em;font-size:.75rem}
  .result .amt{font-family:'DM Serif Display',serif;font-size:2.8rem;color:var(--ink);margin:6px 0 2px}
  .result p{margin:0;color:var(--muted);font-size:.88rem}
  .err{margin-top:20px;padding:14px;border-radius:10px;background:#fdeaea;color:#c53030}
  footer{text-align:center;color:#8f8bbd;font-size:.8rem;margin-top:18px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="badge">🛡️</div>
    <h1>Insurance Cost Predictor</h1>
    <p>Estimate annual medical insurance charges using an XGBoost model.</p>
  </header>
  <div class="card">
    <form method="post" action="/">
      <div class="grid">
        <div><label for="age">Age (years)</label>
          <input id="age" name="age" type="number" min="1" max="120" step="1" required placeholder="e.g. 35" value="{{v.get('age','')}}"></div>
        <div><label for="sex">Sex</label>
          <select id="sex" name="sex">{% for k,t in sex %}<option value="{{k}}" {{'selected' if v.get('sex')|string==k|string}}>{{t}}</option>{% endfor %}</select></div>
        <div><label for="bmi">BMI</label>
          <input id="bmi" name="bmi" type="number" min="10" max="70" step="0.1" required placeholder="e.g. 27.5" value="{{v.get('bmi','')}}"></div>
        <div><label for="children">Children</label>
          <input id="children" name="children" type="number" min="0" max="10" step="1" required placeholder="e.g. 2" value="{{v.get('children','')}}"></div>
        <div><label for="smoker">Smoker</label>
          <select id="smoker" name="smoker">{% for k,t in smoker %}<option value="{{k}}" {{'selected' if v.get('smoker')|string==k|string}}>{{t}}</option>{% endfor %}</select></div>
        <div><label for="region">Region</label>
          <select id="region" name="region">{% for k,t in region %}<option value="{{k}}" {{'selected' if v.get('region')|string==k|string}}>{{t}}</option>{% endfor %}</select></div>
      </div>
      <button type="submit">Predict Charges</button>
    </form>
    {% if error %}<div class="err">{{error}}</div>{% endif %}
    {% if result is not none %}
    <div class="result">
      <small>Estimated annual charges</small>
      <div class="amt">${{ '{:,.2f}'.format(result) }}</div>
      <p>A statistical estimate based on the training data, not an official quote.</p>
    </div>
    {% endif %}
  </div>
  <footer>For educational use only.</footer>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, values = None, None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            row = {f: float(values[f]) for f in FEATURES}
            X = pd.DataFrame([row], columns=FEATURES)
            result = max(0.0, float(model.predict(X)[0]))
        except (KeyError, ValueError):
            error = "Please fill in every field with valid numbers."
    return render_template_string(PAGE, v=values, sex=SEX, smoker=SMOKER,
                                  region=REGION, result=result, error=error)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
