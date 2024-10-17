from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

app = Flask(__name__)

model, tokenizer = None, None

def load_model_and_tokenizer():
    global model, tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("DavinciTech/BERT_Categorizer")
    tokenizer = AutoTokenizer.from_pretrained("DavinciTech/BERT_Categorizer")
    model.to("cpu")

def get_predictions(input_texts):
    id2label = {k: l for k, l in enumerate(model.config.LABEL_DICTIONARY.keys())}
    
    encoded = tokenizer(input_texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to("cpu")
    logits = model(**encoded).logits.cpu().detach().numpy()

    IMPACT_LABELS = ["I1", "I2", "I3", "I4"]
    IMPACT_INDICES = range(0, 4)
    URGENCY_LABELS = ["U1", "U2", "U3", "U4"]
    URGENCY_INDICES = range(4, 8)
    TYPE_LABELS = ["T1", "T2", "T3", "T4", "T5"]
    TYPE_INDICES = range(8, 13)

    def get_preds_from_logits(logits):
        ret = np.zeros(logits.shape)
        best_class = np.argmax(logits[:, IMPACT_INDICES], axis=-1)
        ret[list(range(len(ret))), best_class] = 1
        
        ret[:, URGENCY_INDICES] = 0
        ret[:, TYPE_INDICES] = 0
        
        max_priority_index = np.argmax(logits[:, URGENCY_INDICES], axis=-1)
        ret[list(range(len(ret))), max_priority_index + URGENCY_INDICES[0]] = 1

        max_type_index = np.argmax(logits[:, TYPE_INDICES], axis=-1)
        ret[list(range(len(ret))), max_type_index + TYPE_INDICES[0]] = 1

        return ret

    preds = get_preds_from_logits(logits)
    decoded_preds = [[id2label[i] for i, l in enumerate(row) if l == 1] for row in preds]

    return decoded_preds

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_texts = data['texts']
    predictions = get_predictions(input_texts)
    results = []
    for text, pred in zip(input_texts, predictions):
        results.append({
            'text': text,
            'impact': [model.config.LABEL_DICTIONARY[l] for l in pred if l.startswith("I")],
            'urgency': [model.config.LABEL_DICTIONARY[l] for l in pred if l.startswith("U")],
            'type': [model.config.LABEL_DICTIONARY[l] for l in pred if l.startswith("T")]
        })
    return jsonify(results)

if __name__ == '__main__':
    load_model_and_tokenizer()
    app.run(debug=True)