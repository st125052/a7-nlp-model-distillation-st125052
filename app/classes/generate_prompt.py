import torch

def predict_toxicity(text, model, tokenizer, device, id2word):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    return id2word[predicted_class], probs[0][predicted_class].item()

def get_prediction(input_text, model, tokenizer, device):
    id2word = {
        0: 'Not Toxic', 
        1: 'Toxic', 
        2: 'Not Toxic', 
        3: 'Not Toxic'
    }
    label, confidence = predict_toxicity(input_text, model, tokenizer, device, id2word)
    return label, confidence