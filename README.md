# CSCI544 - Final Project

This project aims to validate the experiment results in the paper [Learning to Generate Question by Asking Question: A Primal-Dual Approach with Uncommon Word Generation](https://aclanthology.org/2022.emnlp-main.4.pdf) due to the loss of source code provided by the authors. We developed the project based on an unofficial implementation [link](https://github.com/Shashwath-kumar/Question-Generation-by-Asking-Questions/tree/main).

## Fine-tuned model

- Fine-tuned model provided in unofficial implementation: [link](https://github.com/Shashwath-kumar/Question-Generation-by-Asking-Questions/releases/download/QG_SQuAD/QG_SQuAD.pt)
- Fine-tuned model by us on SQuAD1.0-split1: [link](https://drive.google.com/file/d/1JIJGUze8l5lL2RWSsdL3s1lwOe2fR2K6/view?usp=sharing)
- Fine-tuned model by us on SQuAD1.0-split2: [link](https://drive.google.com/file/d/1dFkc39CXUQ4qgcxLYcwZgx7JVoxjUGfX/view?usp=sharing)

## Requirements

To install the required dependencies, run:

```py
pip install -r requirements.txt
```

## Usage

1. To train the model, run:
```py
python train.py
```
2. To evaluate the model with pre-trained models, run:
```py
python eval.py
```
3. To perform test on the inference result with BLEU-4, METEOR, and ROUGE scores, run:
```py
python test.py
```

## Reference
```
@inproceedings{wang-etal-2022-learning-generate,
    title = "Learning to Generate Question by Asking Question: A Primal-Dual Approach with Uncommon Word Generation",
    author = "Wang, Qifan  and
      Yang, Li  and
      Quan, Xiaojun  and
      Feng, Fuli  and
      Liu, Dongfang  and
      Xu, Zenglin  and
      Wang, Sinong  and
      Ma, Hao",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.4",
    pages = "46--61",
}
```
