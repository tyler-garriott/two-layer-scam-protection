# two-layer-phishing-protection
COSC569 Project

# DATA USED CAN BE FOUND AT THE LINK BELOW:
Unzip in directory:
[Data](https://drive.google.com/file/d/1VRgL1HccCcgnmbwCRIJ-VETQsdX_XVys/view?usp=sharing).

## Command to install everything you need
pip install pandas numpy scikit-learn xgboost fastapi uvicorn joblib

## ML Model Locally
python3 phish_guard_onefile.py --data data/raw --use-only-malicious --out artifacts --port 8001 --host 0.0.0.0 --no-loso

## Use Modelfile to create llm with ollama
ollama create phish-guard-stage2 -f ollama/Modelfile

## Run Ollama Server Locally
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

## Steps to add the extension to chrome
* Go to chrome://extension
* On the top right there should be a developer mode toggle click it on
* Click load unpacked
* Select extension folder
* The extension should now be working as long as the servers are running locally! Click it to see the ML model dropdown and go to gmail to scan an email!
