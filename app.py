from flask import Flask, render_template, request, session, redirect, url_for, jsonify, Response
import pickle, numpy as np, pandas as pd, os, io, csv, json, uuid, random
from datetime import date

app = Flask(__name__)
app.secret_key = "super_secret_baby_key"

# ── DOCTOR STORE ───────────────────────────────────────────────
# username → full doctor details + password
DOCTORS = {
    "admin": {
        "password":       "baby123",
        "full_name":      "Admin Doctor",
        "age":            "35",
        "dob":            "1990-01-01",
        "mobile":         "9876543210",
        "specialization": "Oncologist",
        "hospital":       "Admin Hospital",
        "created_at":     str(date.today()),
    }
}

# ── MODEL LOADING ──────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
with open(os.path.join(MODEL_DIR, "all_models.pkl"), "rb") as f: ALL_MODELS = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"),     "rb") as f: scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "features.pkl"),   "rb") as f: feature_names = [str(f) for f in pickle.load(f)]
CSV_STORE = {}

# ── AUTO MATCH ─────────────────────────────────────────────────
def normalize(s):
    s = s.strip().lower().replace("_", " ")
    s = s.replace(" se", " error")
    return s

def auto_match(user_cols, required_cols):
    mapping = {}
    used    = set()
    def tokenize(s): return set(normalize(s).split())
    for req in required_cols:
        req_norm = normalize(req)
        for uc in user_cols:
            if uc in used: continue
            if normalize(uc) == req_norm:
                mapping[req] = uc; used.add(uc); break
    for req in required_cols:
        if req in mapping: continue
        req_tokens = tokenize(req)
        best, best_score = None, 0
        for uc in user_cols:
            if uc in used: continue
            uc_tokens = tokenize(uc)
            overlap   = len(req_tokens & uc_tokens)
            union     = len(req_tokens | uc_tokens)
            score     = overlap / union if union else 0
            if score > 0.4 and score > best_score:
                best_score, best = score, uc
        if best: mapping[req] = best; used.add(best)
    return mapping

# ═══════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════════

# ── LOGIN ──────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        pwd  = request.form.get("password", "")

        if user in DOCTORS and DOCTORS[user]["password"] == pwd:
            # Generate OTP
            otp = str(random.randint(1000, 9999))
            session["pending_otp"]  = otp
            session["temp_user"]    = user
            session["doctor_name"]  = DOCTORS[user]["full_name"]
            session["doctor_mobile"]= DOCTORS[user]["mobile"]
            return redirect(url_for("verify_otp"))

        return render_template("login.html", error="Invalid username or password!")
    return render_template("login.html")

# ── OTP VERIFY ─────────────────────────────────────────────────
@app.route("/verify", methods=["GET", "POST"])
def verify_otp():
    # Use doctor's own mobile (from session)
    mobile = session.get("doctor_mobile", "XXXXXXXXXX")
    otp    = session.get("pending_otp", "")

    if request.method == "POST":
        if request.form.get("otp") == otp:
            session["logged_in"] = True
            session["username"]  = session.get("temp_user")
            session.pop("pending_otp", None)
            return redirect(url_for("home"))
        return render_template("verify.html",
            error="Wrong OTP! Try again.",
            mobile=mobile, otp=otp)

    return render_template("verify.html", mobile=mobile, otp=otp)

# ── SIGNUP ─────────────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    today = str(date.today())

    if request.method == "POST":
        username         = request.form.get("username",         "").strip()
        password         = request.form.get("password",         "")
        confirm_password = request.form.get("confirm_password", "")
        full_name        = request.form.get("full_name",        "").strip()
        age              = request.form.get("age",              "").strip()
        dob              = request.form.get("dob",              "").strip()
        mobile           = request.form.get("mobile",           "").strip()
        specialization   = request.form.get("specialization",   "").strip()
        hospital         = request.form.get("hospital",         "").strip()

        # Keep form data to re-fill on error
        form_data = {
            "username": username, "full_name": full_name,
            "age": age, "dob": dob, "mobile": mobile,
            "specialization": specialization, "hospital": hospital,
        }

        # ── Validations ────────────────────────────────────────
        if not username or len(username) < 4:
            return render_template("signup.html",
                error="Username must be at least 4 characters.", form_data=form_data, today=today)

        if username in DOCTORS:
            return render_template("signup.html",
                error=f"Username '{username}' already exists! Please choose another.", form_data=form_data, today=today)

        if len(password) < 8:
            return render_template("signup.html",
                error="Password must be at least 8 characters.", form_data=form_data, today=today)

        if password != confirm_password:
            return render_template("signup.html",
                error="Passwords do not match!", form_data=form_data, today=today)

        if not mobile.isdigit() or len(mobile) != 10:
            return render_template("signup.html",
                error="Mobile number must be exactly 10 digits.", form_data=form_data, today=today)

        if not age.isdigit() or not (22 <= int(age) <= 80):
            return render_template("signup.html",
                error="Age must be between 22 and 80.", form_data=form_data, today=today)

        if not full_name:
            return render_template("signup.html",
                error="Full name is required.", form_data=form_data, today=today)

        if not specialization:
            return render_template("signup.html",
                error="Please select your specialization.", form_data=form_data, today=today)

        # ── Save new doctor ────────────────────────────────────
        DOCTORS[username] = {
            "password":       password,
            "full_name":      full_name,
            "age":            age,
            "dob":            dob,
            "mobile":         mobile,
            "specialization": specialization,
            "hospital":       hospital,
            "created_at":     today,
        }

        # ── Redirect to login after successful signup ──────────
        return redirect(url_for("login", registered="yes"))

    # GET — check if coming from successful registration
    registered = request.args.get("registered")
    return render_template("signup.html", today=today)

# ── LOGOUT ─────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ═══════════════════════════════════════════════════════════════
#  MAIN ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    models_info = {k: {"name": v["name"], "description": v["description"],
        "icon": v["icon"], "color": v["color"], "accuracy": v["accuracy"]}
        for k, v in ALL_MODELS.items()}
    return render_template("home.html", models=models_info)

@app.route("/predict/<model_key>", methods=["GET"])
def predict_page(model_key):
    if not session.get("logged_in"): return redirect(url_for("login"))
    if model_key not in ALL_MODELS:  return redirect(url_for("home"))
    sel = ALL_MODELS[model_key]
    return render_template("predict.html",
        model_key=model_key, model_name=sel["name"],
        model_icon=sel["icon"], model_color=sel["color"],
        model_accuracy=sel["accuracy"], model_description=sel["description"],
        feature_names=feature_names)

@app.route("/predict/<model_key>", methods=["POST"])
def run_predict(model_key):
    if model_key not in ALL_MODELS: return jsonify({"error": "Invalid model"}), 400
    try:
        values = [float(request.form.get(n, 0)) for n in feature_names]
        arr    = np.array(values).reshape(1, -1)
        sel    = ALL_MODELS[model_key]
        X      = scaler.transform(arr) if sel["needs_scale"] else arr
        pred   = sel["model"].predict(X)[0]
        prob   = sel["model"].predict_proba(X)[0]
        if pred == 1:
            result, badge, conf = "Benign", "success", f"{prob[1]*100:.1f}%"
        else:
            result, badge, conf = "Malignant", "danger", f"{prob[0]*100:.1f}%"
        return render_template("result.html",
            result=result, badge=badge, confidence=conf,
            model_name=sel["name"], model_icon=sel["icon"],
            model_color=sel["color"], model_key=model_key,
            all_models={k: {"name": v["name"], "icon": v["icon"],
                "color": v["color"]} for k, v in ALL_MODELS.items()})
    except Exception as e:
        return render_template("result.html", result="Error", badge="warning",
            confidence="N/A", model_name="Error", model_icon="❌",
            model_color="#ccc", model_key=model_key, all_models={})

@app.route("/csv-upload/<model_key>", methods=["POST"])
def csv_upload(model_key):
    try:
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            return jsonify({"error": "Please select a CSV file!"}), 400
        content   = file.read()
        df        = pd.read_csv(io.BytesIO(content))
        user_cols = list(df.columns)
        mapping   = auto_match(user_cols, feature_names)
        unmatched = [f for f in feature_names if f not in mapping]
        store_id  = str(uuid.uuid4())
        CSV_STORE[store_id] = df.to_json(orient="records")
        return jsonify({
            "success": True, "store_id": store_id,
            "user_cols": user_cols, "required_cols": feature_names,
            "auto_mapping": mapping, "unmatched": unmatched,
            "total_rows": len(df), "all_matched": len(unmatched) == 0
        })
    except Exception as e:
        return jsonify({"error": f"CSV read error: {str(e)[:200]}"}), 500

@app.route("/csv-predict/<model_key>", methods=["POST"])
def csv_predict(model_key):
    try:
        mapping  = json.loads(request.form.get("mapping", "{}"))
        store_id = request.form.get("store_id", "").strip()
        if not store_id or store_id not in CSV_STORE:
            return jsonify({"error": "CSV data not found — please re-upload!"}), 400
        df         = pd.read_json(CSV_STORE[store_id])
        rename_map = {v: k for k, v in mapping.items()}
        df         = df.rename(columns=rename_map)
        missing    = []
        for feat in feature_names:
            if feat not in df.columns:
                df[feat] = 0.0; missing.append(feat)
        sel   = ALL_MODELS[model_key]
        X     = df[feature_names].values.astype(float)
        Xi    = scaler.transform(X) if sel["needs_scale"] else X
        preds = sel["model"].predict(Xi)
        probs = sel["model"].predict_proba(Xi)
        results = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            pid = df["id"].iloc[i] if "id" in df.columns \
                  else (df["patient_id"].iloc[i] if "patient_id" in df.columns else i+1)
            results.append({
                "row": i+1, "id": str(pid),
                "prediction": "Benign"    if pred==1 else "Malignant",
                "confidence": f"{prob[1]*100:.1f}%" if pred==1 else f"{prob[0]*100:.1f}%",
                "status":     "success"   if pred==1 else "danger",
            })
        benign = sum(1 for r in results if r["prediction"]=="Benign")
        CSV_STORE.pop(store_id, None)
        return jsonify({"success": True, "results": results,
            "total": len(results), "benign": benign,
            "malignant": len(results)-benign, "missing_features": missing})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)[:300]}"}), 500

@app.route("/sample-csv")
def sample_csv():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df   = pd.DataFrame(data.data[:10], columns=feature_names)
    df.insert(0, "patient_id", [f"P{i+1:03d}" for i in range(10)])
    out  = io.StringIO(); df.to_csv(out, index=False); out.seek(0)
    return Response(out.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_patients.csv"})

@app.route("/download-results", methods=["POST"])
def download_results():
    rows   = json.loads(request.form.get("rows_json", "[]"))
    out    = io.StringIO(); writer = csv.writer(out)
    writer.writerow(["Row", "Patient ID", "Prediction", "Confidence"])
    for r in rows: writer.writerow([r["row"], r.get("id",""), r["prediction"], r["confidence"]])
    out.seek(0)
    return Response(out.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"})

@app.route("/sample/<sample_type>")
def sample(sample_type):
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    idx  = list(data.target).index(1 if sample_type=="benign" else 0)
    return jsonify(dict(zip(feature_names, [float(v) for v in data.data[idx]])))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
