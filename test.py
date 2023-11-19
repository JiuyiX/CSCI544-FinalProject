import csv
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# 初始化两个空列表
hypotheses = []
references = []

# 打开CSV文件
with open('SQuAD_Github_1.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        hypotheses.append(row['hypothesis'])
        references.append([row['references'][2:-2]])

def calculate_bleu(reference_sentences, hypothesis_sentences):
    # Ensure that references and hypotheses are tokenized
    references = [[word_tokenize(ref) for ref in refs] for refs in reference_sentences]
    hypotheses = [word_tokenize(hyp) for hyp in hypothesis_sentences]

    # Calculate BLEU-4 score
    score = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return score

print(calculate_bleu(references, hypotheses))

def calculate_meteor(reference_sentences, hypothesis_sentences):
    # Calculate METEOR for each sentence pair
    scores = [meteor_score([word_tokenize(ref)], word_tokenize(hyp)) for refs, hyp in zip(reference_sentences, hypothesis_sentences) for ref in refs]

    # Average METEOR score across all sentences
    average_score = sum(scores) / len(scores)
    return average_score

print(calculate_meteor(references, hypotheses))

def calculate_rouge(reference_sentences, hypothesis_sentences):
    # Initialize Rouge
    rouge = Rouge()

    # Calculate ROUGE scores for each pair of reference and hypothesis
    rouge = Rouge()
    rouge_scores = [rouge.get_scores(hyp, ref[0]) for hyp, ref in zip(hypothesis_sentences, reference_sentences)]
    # Extract and average the ROUGE-L scores (or ROUGE-1, ROUGE-2 as needed)
    average_score = sum([score[0]['rouge-l']['f'] for score in rouge_scores]) / len(rouge_scores)
    average_score_1 = sum([score[0]['rouge-l']['p'] for score in rouge_scores]) / len(rouge_scores)
    average_score_2 = sum([score[0]['rouge-l']['r'] for score in rouge_scores]) / len(rouge_scores)

    return average_score, average_score_1, average_score_2

print(calculate_rouge(references, hypotheses))