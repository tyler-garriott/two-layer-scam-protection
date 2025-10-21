# two-layer-phishing-protection
COSC569 Project

# DATA USED CAN BE FOUND AT THE LINK BELOW:
Unzip in directory:
[Data](https://drive.google.com/file/d/1VRgL1HccCcgnmbwCRIJ-VETQsdX_XVys/view?usp=sharing).

## Command to install everything you need
pip install pandas numpy scikit-learn xgboost fastapi uvicorn joblib

## Live API
python phish_guard_onefile.py --data data/raw \
  --out artifacts \
  --url-col url --label-col type --drop-labels "defacement,malware" \
  --port 8000

### Example Command for Live API
curl -s -X POST http://127.0.0.1:8000/scan-email \
  -H "Content-Type: application/json" \
  -d '{"raw_email": "Suspicious login detected", "urls": ["http://phish-login.example.com"]}'

## For offline testing use:
python offline.py

