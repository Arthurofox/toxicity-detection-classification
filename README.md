# toxicity detection project

## overview
this repo contains a machine learning system for toxicity detection in text using the google jigsaw dataset. the project involves building, evaluating, and deploying a toxicity classifier with considerations for fairness and robustness against obfuscation.

## project structure
```
toxicity-detection/
├── data/               # raw and processed data
├── models/             # saved model checkpoints 
├── src/                # source code
│   ├── preprocessing/  # data cleaning and preparation
│   ├── training/       # model training scripts
│   ├── evaluation/     # metrics and evaluation code
│   ├── inference/      # inference scripts
│   └── utils/          # helper functions
├── notebooks/          # jupyter notebooks (for later)
├── tests/              # unit tests
├── configs/            # configuration files
└── requirements.txt    # dependencies
```

## development workflow

1. **data acquisition & exploration** - get jigsaw dataset, analyze distributions
2. **metric definition** - implement classification metrics + fairness metrics
3. **preprocessing pipeline** - tokenization, splits, dataloaders
4. **llm baseline** - sample test set, run inference with open-source llm
5. **roberta finetuning** - training script w/ hyperparameter optimization
6. **model card** - document model details, performance, limitations
7. **bias assessment** - measure performance across demographic subgroups
8. **jailbreak testing** - test obfuscation techniques, build defenses
9. **api creation** - fastapi wrapper with obfuscation defenses
10. **report generation** - visualizations and technical report

## getting started

```bash
# clone repo
git clone https://github.com/Arthurofox/toxicity-detection-classification.git
cd toxicity-detection-classification

# set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on windows
pip install -r requirements.txt

# download data
python src/utils/download_data.py
```

## team
- arthur
- roumouz

## todo
- [ ] implement data preprocessing
- [ ] create llm baseline
- [ ] finetune roberta
- [ ] evaluate fairness
- [ ] test obfuscation resilience
- [ ] api deployment
- [ ] technical report
