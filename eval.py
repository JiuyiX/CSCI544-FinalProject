import sacrebleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import torch
from inference import BeamSearch
from torch.utils.data import DataLoader
from dataset_utils import get_dataset
from final_model import QuestionGenerationModel
import config
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
from nltk.tokenize import word_tokenize

def evaluate(model, tokenizer, device, val_dataloader: DataLoader, beam_size=10, max_len=50) -> None:
    beam_search = BeamSearch(model, tokenizer, device, beam_size=beam_size, max_len=max_len)

    references = []
    hypotheses = []

    for batch in tqdm(val_dataloader):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                
        # Generate the question using Beam Search
        generated_question = beam_search.search(batch)

        # Append the generated question to the list of hypotheses
        hypotheses.append(generated_question)

        # Append the true question to the list of references
        true_question = tokenizer.decode(batch['question_input_ids'][0], skip_special_tokens=True)
        references.append(true_question)

    df = pd.DataFrame({'hypothesis':hypotheses, 'references':references})
    df.to_csv('SQuAD_Bart_base_1.csv', index=False)
    
    # bleu = sacrebleu.corpus_bleu(hypotheses, references)

    # print(f"Evaluation result: {bleu}")
    # bleu_score = bleu.score
    # print(bleu_score)

    # meteor_scores = [meteor_score([word_tokenize(ref[2:-2])], word_tokenize(hyp)) for ref, hyp in zip(references, hypotheses)]
    # # Calculate the average METEOR score for the entire dataset
    # meteor_score1 = float(sum(meteor_scores)) / len(meteor_scores)
    # print(meteor_score1)

    # rouge = Rouge()
    # rouge_scores = [rouge.get_scores(hyp, ref) for hyp, ref in zip(hypotheses, references)]
    # # Extract and average the ROUGE-L scores (or ROUGE-1, ROUGE-2 as needed)
    # rouge_score = sum([score[0]['rouge-l']['f'] for score in rouge_scores]) / len(rouge_scores)
    # rouge_score_1 = sum([score[0]['rouge-l']['p'] for score in rouge_scores]) / len(rouge_scores)
    # rouge_score_2 = sum([score[0]['rouge-l']['r'] for score in rouge_scores]) / len(rouge_scores)
    # print(rouge_score)
    # print(rouge_score_1)
    # print(rouge_score_2)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = QuestionGenerationModel(config.model_name, config.d_model, device)
    model.load_state_dict(torch.load(f'QG_SQuAD_1_BART_base.pt'))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, model_max_length = 512)

    _, _, test_dataloader = get_dataset(bsize=config.batch_size)

    # Evaluate the model on the validation dataset
    evaluate(model, tokenizer, device, test_dataloader)
    print("Evaluation finished!")
