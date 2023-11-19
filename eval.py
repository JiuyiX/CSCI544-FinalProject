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
