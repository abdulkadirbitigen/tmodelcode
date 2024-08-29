import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split

# BERT Tokenizer'ı yükleyin
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# JSONL veri setini yükleme
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)


def preprocess_data(data, label_map):
    texts = []
    labels = []
    for entry in data:
        text = entry['text']
        label = entry['label']

        # Texti boşluklara göre tokenize et
        word_positions = []
        start = 0
        for word in text.split():
            end = start + len(word)
            word_positions.append((start, end))
            start = end + 1  # 1 boşluk karakteri için

        encoded_labels = encode_labels_for_words(label, word_positions, label_map)

        texts.append(text)
        labels.append(encoded_labels)
    return texts, labels


# Kelime bazlı etiketlemeyi kodlama
def encode_labels_for_words(labels, word_positions, label_map):
    encoded_labels = [[0] * len(label_map) for _ in range(len(word_positions))]

    for start, end, label in labels:
        for i, (word_start, word_end) in enumerate(word_positions):
            if start < word_end and end > word_start:  # Kelime ve etiket kesişiyorsa
                encoded_labels[i][label_map[label]] = 1
    return encoded_labels


# Manuel olarak girilen label_map
label_map = {
    'ANAT': 0,
    'OBS-PRESENT': 1,
    'OBS-ABSENT': 2,
    'OBS-UNCERTAIN': 3,
    'IMPRESSION': 4
}

# Dört veri setini yükleyin
data1 = list(load_jsonl('all.jsonl'))
data2 = list(load_jsonl('all1.jsonl'))
data3 = list(load_jsonl('all2.jsonl'))
data4 = list(load_jsonl('all3.jsonl'))

# Veri setlerini birleştirin
combined_data = data1 + data2 + data3 + data4

# Birleştirilmiş veri setini işleyin
texts, labels = preprocess_data(combined_data, label_map)


def tokenize_and_encode(texts, labels):
    input_ids = []
    attention_masks = []
    encoded_labels = []

    for text, label in zip(texts, labels):
        encoded = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='tf')
        input_ids.append(encoded['input_ids'][0].numpy())
        attention_masks.append(encoded['attention_mask'][0].numpy())
        padded_labels = label + [[0] * len(label_map)] * (512 - len(label))  # Padding için sıfırları ekleyin
        encoded_labels.append(padded_labels)

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_masks), tf.convert_to_tensor(encoded_labels)


train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1)

train_input_ids, train_attention_masks, train_encoded_labels = tokenize_and_encode(train_texts, train_labels)
test_input_ids, test_attention_masks, test_encoded_labels = tokenize_and_encode(test_texts, test_labels)


# Modeli Tanımlama
class CustomBertModel(tf.keras.Model):
    def __init__(self, num_labels):
        super(CustomBertModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.classifier = tf.keras.layers.Dense(num_labels, activation='sigmoid')

    def call(self, inputs, attention_mask=None, training=False):
        outputs = self.bert(inputs, attention_mask=attention_mask, training=training)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits


num_labels = len(label_map)
model = CustomBertModel(num_labels)

# Modeli derleme
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
history = model.fit(
    [train_input_ids, train_attention_masks],
    train_encoded_labels,
    validation_split=0.1,
    epochs=20,
    batch_size=1
)

# Modeli kaydetme
model.save('teknomodel', save_format='tf')  # SavedModel formatında kaydedin

