import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np

# Label map'i tanımlama
label_map = {
    'ANAT': 0,
    'OBS-PRESENT': 1,
    'OBS-ABSENT': 2,
    'OBS-UNCERTAIN': 3,
    'IMPRESSION': 4
}

# Reverse label map
reverse_label_map = {v: k for k, v in label_map.items()}


# CustomBertModel sınıfını tanımlama
class CustomBertModel(tf.keras.Model):
    def __init__(self, num_labels):
        super(CustomBertModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.classifier = tf.keras.layers.Dense(num_labels, activation='softmax')

    def call(self, inputs, attention_mask=None, training=False):
        outputs = self.bert(inputs, attention_mask=attention_mask, training=training)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits


# Model ve tokenizer'ı yükleme
model = tf.keras.models.load_model('teknomodel', custom_objects={'CustomBertModel': CustomBertModel})
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def predict_labels(text):
    # Tokenizasyon
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Model ile tahmin yapma
    logits = model(input_ids, attention_mask=attention_mask)

    # Etiketlerin logits değerlerini numpy dizisine dönüştürme
    logits = logits.numpy()

    # Tahminleri en yüksek olasılığa göre seçme
    predictions = np.argmax(logits, axis=-1).flatten()

    # Çıktı düzenleme
    tokens = tokenizer.convert_ids_to_tokens(input_ids.numpy().flatten())
    results = []

    # Kelime kelime ayırma ve etiketleme
    current_word = ""
    current_label = None
    for i, (token, prediction) in enumerate(zip(tokens, predictions)):
        if token in tokenizer.convert_ids_to_tokens([tokenizer.pad_token_id]):
            continue

        if token.startswith('##'):
            token = token[2:]  # Subword tokenları birleştir
            current_word += token
        else:
            if current_word and current_label in label_map.values():
                results.append((current_word, reverse_label_map.get(current_label, "UNKNOWN")))
            current_word = token
            current_label = prediction

    if current_word and current_label in label_map.values():
        results.append((current_word, reverse_label_map.get(current_label, "UNKNOWN")))

    return [result for result in results if result[1] != "UNKNOWN"]


# Kullanıcıdan metin alma
user_input = input("Metin girin: ")
labels = predict_labels(user_input)

# Etiketleri ekrana yazdırma
for token, label in labels:
    print(f"{token} = {label}")

