from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import base64
import hashlib

app = Flask(__name__)
CORS(app)

# Blockchain simulation
blockchain = []

# Random Forest training data (dummy for demo)
X_train = [
    [1000, 1, 0], [5000, 5, 1], [2000, 2, 0], [8000, 8, 3], [3000, 3, 0]
]  # [claim_amount, policy_age, previous_claims]
y_train = [0,1,0,1,0]  # 0=low fraud, 1=high fraud

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Helper: convert cv2 image to base64
def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

# Blockchain helper
def add_block(claim_status, fraud_score, insurance_type, claim_amount):
    prev_hash = blockchain[-1]['block_hash'] if blockchain else '0'*64
    block_data = f"{claim_status}{fraud_score}{insurance_type}{claim_amount}{prev_hash}"
    block_hash = hashlib.sha256(block_data.encode()).hexdigest()
    block = {
        'block_index': len(blockchain)+1,
        'claim_status': claim_status,
        'fraud_score': fraud_score,
        'insurance_type': insurance_type,
        'claim_amount': claim_amount,
        'prev_hash': prev_hash,
        'block_hash': block_hash
    }
    blockchain.append(block)
    return block

# Damage detection for Vehicle/Crop
def detect_damage(file_bytes, insurance_type):
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 70, 200)  # stricter threshold

    # Highlight edges for frontend
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (0,0,255), 2)
    highlight = cv2.addWeighted(img, 0.8, mask, 0.8, 0)

    damage_detected = False

    if insurance_type == "Vehicle":
        # Only consider large areas
        large_contours = [c for c in contours if cv2.contourArea(c) > 200]  # major damage threshold
        if len(large_contours) >= 2:
            damage_detected = True
    elif insurance_type == "Crop":
        area = cv2.countNonZero(edges)
        if area > 5000:  # ignore neat crops
            damage_detected = True

    return damage_detected, image_to_base64(highlight)

@app.route('/verify', methods=['POST'])
def verify_claim():
    insurance_type = request.form.get('insurance_type')
    claim_amount = float(request.form.get('claim_amount',0))
    policy_age = float(request.form.get('policy_age',1))
    prev_claims = float(request.form.get('prev_claims',0))
    
    damage_found = False
    highlighted_image = None

    if insurance_type in ['Vehicle','Crop']:
        file = request.files.get('file')
        if not file:
            return jsonify({'error':'File required for Vehicle/Crop claims'}),400
        damage_found, highlighted_image = detect_damage(file.read(), insurance_type)

    # Fraud risk prediction
    X_input = [[claim_amount, policy_age, prev_claims]]
    fraud_pred = rf_model.predict_proba(X_input)[0][1]*100  # probability of fraud
    fraud_score = int(fraud_pred)

    # Claim decision rules
    if insurance_type == 'Health':
        claim_status = "Approved"
    elif not damage_found:
        claim_status = "Rejected"
    else:
        # Major damage + fraud check
        claim_status = "Approved" if fraud_score < 80 else "Rejected"

    block = add_block(claim_status, fraud_score, insurance_type, claim_amount)

    return jsonify({
        'claim_status': claim_status,
        'fraud_score': fraud_score,
        'highlighted_image': highlighted_image,
        'accident_analysis': 'Damage detected' if damage_found else 'No damage detected',
        'block': block
    })

@app.route('/blockchain', methods=['GET'])
def get_blockchain():
    return jsonify(blockchain)

if __name__ == "__main__":
    app.run(debug=True)