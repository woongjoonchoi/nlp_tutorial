import config
import torch

import flask
from flask import Flask
from flask import request
from model import BERTBaseUncased


import torch.nn as nn

app = Flask(__name__)

MODEL = None
DEVICE = "cuda"


def sentence_prediction(sentence) :
    tokenizer = config.TOKENIZER
    max_length = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_length,
        pad_to_max_length=True,
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]


    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)


    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)



#  output type변경해줘야함 cpu().detach().numpy()
    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]
@app.route("/predict")
def predict() :
    sentence = request.args.get("sentence")
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1- positive_prediction

    # print(sentence)
    response = {}

    #TypeError: Object of type Tensor is not JSON serializable   type change str
    response["response"] = {
        "positive" : str(positive_prediction),
        "negative" : str(negative_prediction),
        "sentence" : str(sentence)
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    # MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()






