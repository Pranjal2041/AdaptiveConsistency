Model outputs from different models on 13 different datasets. 

Run `bash download_outputs.sh` to download all the outputs. \
<br>

Directory structure:
```
outputs
├── README.md
├── code-davinci-002
│   ├── dataset-1
│   │   ├── outputs_seed1.jsonl
│   │   ├── outputs_seed2.jsonl
│   │   └── outputs_seed3.jsonl
│   ├── dataset-2
│   │   ├── ...
│   ├── ...
├── vicuna-13b
│   ├── ...
└── ...
```

<br>

We use the following hyperparameters for model generation:
- `temperature`: 0.7
- `top_p`: 1.0
- `max_length`: 512
- `max_gens`: 40

