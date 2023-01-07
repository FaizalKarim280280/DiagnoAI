# DiagnoAI
DiagnoAI is a tool to detect a disease from a text description of the patient's symptoms and daily condition. It is based on a transformer model called BERT, fine-tuned for 24 common diseases.

# Contents
1. Dataset
2. Model Training
3. Testing
4. References

# 1. Dataset

We created a dataset containing 24 disease and 50 manually written descriptions of the symptoms (in english) for each disease. The disease names, symptoms and precautions where chosen Disease Symptom Prediction dataset [1] from Kaggle.

Hence, a total 1200 descriptions were created, out of which 80% was used for model training and remaining 20% for validation and testing purposes. An example of a data instance:

<pre><code>Description : There are small red spots all over my body that I can't explain. It's worrying me. I feel extremely tired and experience a mild fever every night.

Disease: Chicken Pox
</code></pre>

# 2. Model Training

Because of limited data, we decided to fine tune a pretrained language model. We chose the pre trained BERT model from Hugging Face and its corresponding tokenizer for tokenizing the sentences. TensorFlow was used as the base framework for loading and training the model.

Upon experimentation, we found that unfreezing the BERT layer helped acheive a better training and validation accuracy. Hence, we decided to go keep the complete model trainable.

The model was trained with the following parameters:

Loss function: SparseCategoricalCrossentropy
Optimizer: Adam
Learning Rate: 0.00003
Epochs: 5

![Model Plot](/static/src/model_plot.png)

# 3. Testing

After training, we acheived a training accuracy of 100.00% and vadlidation accuracy of 98.33%. Although, the misclassification rate is quite low, we can't be completely sure of the model's predictions as it trained on a relatively smaller corpus.

We plan to increase the dataset in future so that the model can generalize better and not suffer from overconfidence.

# 4. References

1. [Kaggle - Disease Symptom Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
2. [Hugging Face - BERT](https://huggingface.co/docs/transformers/model_doc/bert)
3. [Tensorflow](https://www.tensorflow.org/)
4. [Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding."](https://arxiv.org/abs/1810.04805)