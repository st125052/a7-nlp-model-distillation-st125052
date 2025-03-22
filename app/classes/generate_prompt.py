import torch
import pickle

def get_torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_model_and_tokenizer(device):
    with open("../models/final-model/student_model.pkl", "rb") as f:
        model = pickle.load(f).to(device)
    with open("../models/final-model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def predict_toxicity(text, model, tokenizer, device, id2word):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    return id2word[predicted_class], probs[0][predicted_class].item()

def get_prediction(input_text):
    device = get_torch_device()
    model, tokenizer = get_model_and_tokenizer(device)
    id2word = {
        0: 'Not Toxic', 
        1: 'Toxic', 
        2: 'Not Toxic', 
        3: 'Not Toxic'
    }
    label, confidence = predict_toxicity(input_text, model, tokenizer, device, id2word)
    return label, confidence