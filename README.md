# Context Collection Competition Starter Kit

Welcome to the starter kit for the [Context Collection Competition](https://jb.gg/co4). 
It will guide you from the moment of downloading data to running a baseline solution and making your first submission.
Please feel free to fork the starter kit to use it as the starting point for your solutions.

### Objective
The objective of the competition is to implement a context collection strategy that yields the most accurate code completions when provided to LLMs.
Please read the [competition page](https://jb.gg/co4) for details.

## Getting Started 

### Data Preparation

The starter kit expects that data files are stored in the `data` folder. 
`stage` is the competition stage, for the files already available in the starter kit `stage = "start"`. 
`language` is the language split, `kotlin` or `python`.

The structure for data is as follows:
```bash
data
├── {language}-{stage}.jsonl # Competition data
└── repositories-{language}-{stage} # Folder with repositories
    └── {owner}__{repository}-{revision} # Repository revision used for collecting context
        └── repository contents
```

To prepare data for the starter kit:
1. Download the data for the respective stage from [the shared folder](https://drive.google.com/drive/folders/1wcpq7ob33z5wHNFzUaiJWuHWw8sNuumC). Please note: unpacked data takes ~10GB on disk.
2. Put the `{language}-{stage}.jsonl` file (datapoints) and the `{language}-{stage}.zip` archive (repos) to the `data` folder.
3. Run `./prepare_data.sh practice python`, possibly replacing `practice` with the stage and `python` with `kotlin`.


### Running the baselines

The starter kit contains two baselines in [baselines.py](baselines.py) and an option to modify prefix/suffix of the completion file. 
1. Selecting a random Python file from the repository.
2. Selecting a single Python file according to the [BM-25](https://en.wikipedia.org/wiki/Okapi_BM25) metric. 



To run the baselines:
1. `poetry install --no-root` &ndash; install dependencies via poetry
2. `poetry run python baselines.py --stage practice --strategy random --lang python` &ndash; run the baselines
   - You can replace `practice` with `public` to generate a complete submission for the competition
   - You can replace `random` with another strategy, e.g., `bm25`
   - You can replace `python` with `kotlin` for another split
   - You can provide `--trim-prefix` and/or `trim-suffix` to modify the used prefix and suffix of the completion file by trimming it to 10 lines
3. The prediction file will be saved in the `predictions` folder.

### Implementing your own strategy
Please look at the implementation of the baselines in [baselines.py](baselines.py) for an example.
If the selected context contains multiple files, their parts included in the context should be separated by `<|file_sep|>`.

### Prediction file format
The predictions are expected in a JSON Lines file, with each object having a ``context`` field:
```
{"context": "**context for prediction 1**"}
{"context": "**context for prediction 2**"}
...
{"context": "**context for prediction N**"}
```
The number and order of the objects should correspond to the objects in the input ``.jsonl`` file.

By default, the entire prefix and suffix of the file will be provided to the model.
Optionally, you can submit your own version of the prefix and suffix in the completion file.  
In that case, the format of an entry in the file is:
```
{"context": "context for prediction", "prefix": "custom prefix", "suffix": "custom suffix"}
```
Both `prefix` and `suffix` are optional in each entry: it is acceptable if each of the fields is only specified in some of the entries. 

### Submitting your solution

Go to the [competition page](https://eval.ai/web/challenges/challenge-page/2516), enter the competition, select the stage, and upload the generated prediction file.

### Evaluation
The contexts are used to generate completions with three different models based on their similarity to the reference text with the ChrF score metric. 
The final score is the average across the scores of the three models.
Please read the [Evaluation](https://eval.ai/web/challenges/challenge-page/2516/evaluation) section of the competition page for details.

# Private Phase Evaluation
## Build the container
`docker build -t ase2025-imranur .`

## Run with code-chunk strategy
The current working directory should have a `/data` folder where the private instance for python and kotlin is extracted using `./prepare_data.sh private python` and `./prepare_data.sh private kotlin`.

With the `-v` flag,`$(pwd)/data:/app/data` and `$(pwd)/predictions:/app/predictions` mount the external directory (before `:`) to the container (after `:`).

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/predictions:/app/predictions ase2025-imranur poetry run python baselines.py --stage private --strategy code-chunk --lang python
```

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/predictions:/app/predictions ase2025-imranur poetry run python baselines.py --stage private --strategy code-chunk --lang kotlin
```

