"""Flask server for Vicuna-13b, returns results in OpenAI format."""

from flask import Flask, request, Response, stream_with_context, jsonify
from fastchat.serve.cli import load_model, generate_stream
import time
import torch
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from transformers import StoppingCriteriaList, MaxLengthCriteria, StoppingCriteria


LOG_FILE = "api_requests.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        TimedRotatingFileHandler(LOG_FILE, when="D", interval=1, backupCount=30),
        logging.StreamHandler(),
    ],
)

app = Flask(__name__)

# Load the model and tokenizer
model_name = "vicuna-13b"
device = "cuda"
num_gpus = "4"
load_8bit = False
debug = False
model, tokenizer = load_model(model_name, device, num_gpus, load_8bit, debug)


# Adapted from https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/7
class CustomStopTokenCriteria(StoppingCriteria):
    def __init__(self, stops=[], len_input_ids=0, encounters=1):
        super().__init__()
        self.stops = stops
        self.len_input_ids = len_input_ids
        self.previous_len = len_input_ids
        self.min_stop_token_len = min([len(tokenizer.encode(stop)) for stop in stops])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        len_input_ids = len(input_ids[0])

        # save some time by not checking for stop tokens until we've generated enough tokens.
        # this is not a perfect solution, but it's a good enough heuristic for now.
        if len_input_ids - self.previous_len < self.min_stop_token_len:
            return False
        self.previous_len = len_input_ids

        generated_tokens = input_ids[0][self.len_input_ids :]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        for stop in self.stops:
            if stop in generated_text:
                return True
        return False


@torch.inference_mode()
def generate_text(
    prompt, temperature=0.7, max_new_tokens=150, n=1, stop=None, top_p=0.9
):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    if stop:
        stopping_criteria = StoppingCriteriaList(
            [CustomStopTokenCriteria(stops=stop, len_input_ids=len(input_ids[0]))]
        )

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=n,
        top_p=top_p,
        do_sample=True if (n > 1 or temperature > 0) else False,
        no_repeat_ngram_size=0 if top_p is not None else None,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria if stop else None,
    )

    choices = []
    for seq in output:
        completion = tokenizer.decode(
            seq[len(input_ids[0]) :], skip_special_tokens=True
        )
        stop_token_present = None
        for stop_token in stop:
            if stop_token in completion:
                stop_token_present = stop_token
                break
        finish_reason = "stop_token" if stop and stop_token_present else "length"
        if stop_token_present:
            completion = completion.split(stop_token_present)[0]

        choices.append(
            {
                "text": completion,
                "index": len(choices),
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        )

    response = {
        "id": f"cmpl-{time.time()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(input_ids[0]),
            "completion_tokens": sum(
                [len(tokenizer.encode(choice["text"])) for choice in choices]
            ),
            "total_tokens": len(input_ids[0])
            + sum([len(tokenizer.encode(choice["text"])) for choice in choices]),
        },
    }

    return response


@app.route("/completion", methods=["POST"])
def completion():
    data = request.get_json()
    prompt = data.get("prompt")
    temperature = float(data.get("temperature", 0.7))
    max_new_tokens = int(data.get("max_tokens", 150))
    n = int(data.get("n", 1))
    stop = data.get("stop")
    top_p = data.get("top_p")

    if top_p is not None:
        top_p = float(top_p)

    response = generate_text(prompt, temperature, max_new_tokens, n, stop, top_p)
    output_str = "\n".join([choice["text"] for choice in response["choices"]])
    log_entry = f"Input: {prompt}, Output: {output_str.strip()}, Params: temperature={temperature}, completion_tokens={response['usage']['completion_tokens']}, n={n}, stop={stop}, top_p={top_p}"
    logging.info(log_entry)
    return jsonify(response)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("input")
    temperature = float(data.get("temperature", 0.7))
    max_new_tokens = int(data.get("max_new_tokens", 512))

    def generate_response():
        params = {
            "prompt": user_input,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": tokenizer.eos_token,  # Set the stop parameter to the tokenizer's EOS token
        }
        for response in generate_stream(model, tokenizer, params, device):
            yield response + "\n"

    return Response(stream_with_context(generate_response()), content_type="text/plain")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
