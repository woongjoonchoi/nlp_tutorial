import transformers

DEVICE = "cuda"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS=10
BERT_PATH = "/opt/ml/nlp_tutorial/bert_base_uncased"
MODEL_PATH = "/opt/ml/nlp_tutorial/bert-sentiment/model.bin"
TRAINING_FILE = "/opt/ml/nlp_tutorial/bert_base_uncased/IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH ,do_lower_case = True)