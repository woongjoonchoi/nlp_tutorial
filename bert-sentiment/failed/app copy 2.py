
##   http://127.0.0.1:4016/predict    {"response":{}}


#http://127.0.0.1:4016/predict?sentence=%20i%20love%20this%20video  
import config
import torch

import flask
from flask import Flask
from flask import request
from model import BERTBaseUncased


app = Flask(__name__)

MODEL = None
DEVICE = "cuda"


def sentence_prediction(sentence) :
    tokenizer = config.TOKENIZER
    max_length = config.MAX_LENGTH
    review = str(self.review[item])
    review = " ".join(review.split())

    inputs = self.tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=self.max_len,
        pad_to_max_length=True,
    )

    ids = inputs["input_ids"].unsqueeze(0)
    mask = inputs["attention_mask"].unsqueeze(0)
    token_type_ids = inputs["token_type_ids"].unsqueeze(0)

    ids = ids.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    targets = targets.to(device, dtype=torch.float)

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs)
    return outputs[0][0]
@app.route("/predict")
def predict() :
    sentence = request.args.get("sentence")
    print(sentence)
    response = {}
    response["response"] = {}
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run()






