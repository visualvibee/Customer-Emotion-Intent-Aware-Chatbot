
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = "/content/drive/MyDrive/CustomerSupportAI"

# Load Intent Model
intent_tokenizer = AutoTokenizer.from_pretrained(f"{base_path}/intent_model")
intent_model = AutoModelForSequenceClassification.from_pretrained(
    f"{base_path}/intent_model"
).to(device)
intent_model.eval()

intent_label_encoder = torch.load(
    f"{base_path}/intent_label_encoder.pt",
    weights_only=False
)

# Load Emotion Model
emotion_tokenizer = AutoTokenizer.from_pretrained(f"{base_path}/emotion_model")
emotion_model = AutoModelForSequenceClassification.from_pretrained(
    f"{base_path}/emotion_model"
).to(device)
emotion_model.eval()

emotion_label_encoder = torch.load(
    f"{base_path}/emotion_label_encoder.pt",
    weights_only=False
)

emotion_prefix = {
    "angry": "I completely understand your frustration. ",
    "frustrated": "I can see why this would be frustrating. ",
    "sad": "I'm really sorry you're experiencing this. ",
    "confused": "I understand this might be confusing. ",
    "neutral": "",
    "happy": ""
}

intent_base_response = {
    "complaint": "Let me fix this for you right away.",
    "refund_replace": "I'll check your refund status immediately and update you.",
    "support_request": "I'll guide you step by step to resolve this.",
    "inquiry": "Here’s the information you requested.",
    "appreciation": "We truly appreciate your support!",
    "feedback": "Thank you for helping us improve."
}

def predict(model, tokenizer, label_encoder, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_id].item()

    label = label_encoder.inverse_transform([pred_id])[0]

    return label, round(confidence, 4)

def customer_support_ai(text):

    intent, intent_conf = predict(
        intent_model,
        intent_tokenizer,
        intent_label_encoder,
        text
    )

    emotion, emotion_conf = predict(
        emotion_model,
        emotion_tokenizer,
        emotion_label_encoder,
        text
    )

    prefix = emotion_prefix.get(emotion, "")
    base = intent_base_response.get(intent, "Let me assist you with that.")

    response = prefix + base

    return {
        "intent": intent,
        "intent_confidence": intent_conf,
        "emotion": emotion,
        "emotion_confidence": emotion_conf,
        "response": response
    }
