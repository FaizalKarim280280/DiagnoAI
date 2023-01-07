from tensorflow import keras
from transformers import AutoTokenizer
from transformers import TextClassificationPipeline
from transformers import TFAutoModelForSequenceClassification


class Model:

    def __init__(self, label2id=None, id2label=None):
        if label2id is None:
            self.label2id = {'Psoriasis': 0,
                             'Varicose Veins': 1,
                             'Typhoid': 2,
                             'Chicken pox': 3,
                             'Impetigo': 4,
                             'Dengue': 5,
                             'Fungal infection': 6,
                             'Common Cold': 7,
                             'Pneumonia': 8,
                             'Dimorphic Hemorrhoids': 9,
                             'Arthritis': 10,
                             'Acne': 11,
                             'Urinary Tract Infection': 12,
                             'Allergy': 13,
                             'Gastroesophageal Reflux': 14,
                             'Drug Reaction': 15,
                             'Peptic Ulcer': 16,
                             'Diabetes': 17,
                             'Bronchial Asthma': 18,
                             'Hypertension': 19,
                             'Migraine': 20,
                             'Cervical spondylosis': 21,
                             'Jaundice': 22,
                             'Malaria': 23}
        if id2label is None:
            self.id2label = {0: 'Psoriasis',
                             1: 'Varicose Veins',
                             2: 'Typhoid',
                             3: 'Chicken pox',
                             4: 'Impetigo',
                             5: 'Dengue',
                             6: 'Fungal infection',
                             7: 'Common Cold',
                             8: 'Pneumonia',
                             9: 'Dimorphic Hemorrhoids',
                             10: 'Arthritis',
                             11: 'Acne',
                             12: 'Urinary Tract Infection',
                             13: 'Allergy',
                             14: 'Gastroesophageal Reflux',
                             15: 'Drug Reaction',
                             16: 'Peptic Ulcer',
                             17: 'Diabetes',
                             18: 'Bronchial Asthma',
                             19: 'Hypertension',
                             20: 'Migraine',
                             21: 'Cervical spondylosis',
                             22: 'Jaundice',
                             23: 'Malaria'}
        self.num_classes = len(self.label2id)

    @staticmethod
    def tokenizer():
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        return tokenizer

    def model(self):
        model = TFAutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased",
            num_labels=self.num_classes,
            label2id=self.label2id,
            id2label=self.id2label)

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(3e-5),
            metrics=['accuracy']
        )

        model.load_weights("model-weights/model-weights.h5")

        return model


class Predict:
    
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.num_classes = 24
        
    def predict_disease(self, text):
        pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, top_k=self.num_classes)
        pred = pipe(text)
        return pred
