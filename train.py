import pandas as pd
import torch
import torch.nn as nn
from dataset_utils import get_dataset
from final_model import QuestionGenerationModel
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import config
from transformers import AutoTokenizer
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from inference import BeamSearch
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize

def train_model(model, train_dataloader, val_dataloader, device, num_epochs, learning_rate):
    # Move the model to the device
    model = model.to(device)
    # Set up the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    ce_loss = nn.CrossEntropyLoss()

    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = 5000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            logits, start_scores, end_scores, _, _, y_en, y_pre = model(**batch)

            target_ids = batch['question_input_ids'][:, 1:]
            qg_loss = ce_loss(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            answer_start = batch["token_start"]
            answer_end = batch["token_end"]

            qa_loss = ce_loss(start_scores, answer_start) + ce_loss(end_scores, answer_end)

            kd_loss = -torch.sum(y_en * torch.log(y_pre))

            total_loss = qg_loss + 0.8 * qa_loss + 0.15 * kd_loss
            total_loss.backward()

            optimizer.step()
            scheduler.step()
            train_loss += total_loss.item()

        train_loss /= len(train_dataloader)
        print(f"Train Loss: {train_loss}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

                logits, start_scores, end_scores, _, _, y_en, y_pre = model(**batch)

                target_ids = batch['question_input_ids'][:, 1:]
                qg_loss = ce_loss(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

                answer_start = batch["token_start"]
                answer_end = batch["token_end"]
                qa_loss = ce_loss(start_scores, answer_start) + ce_loss(end_scores, answer_end)

                kd_loss = -torch.sum(y_en * torch.log(y_pre))

                total_loss = qg_loss + 0.8 * qa_loss + 0.15 * kd_loss
                val_loss += total_loss.item()

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config.model_save_name}")
            print(f"Best model saved! {config.model_save_name}")

    return best_model

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

    bleu = sacrebleu.corpus_bleu(hypotheses, references)

    print(f"Evaluation result: {bleu}")
    bleu_score = bleu.score
    print(bleu_score)

    meteor_scores = [meteor_score([word_tokenize(ref[2:-2])], word_tokenize(hyp)) for ref, hyp in zip(references, hypotheses)]
    # Calculate the average METEOR score for the entire dataset
    meteor_score1 = float(sum(meteor_scores)) / len(meteor_scores)
    print(meteor_score1)

    rouge = Rouge()
    rouge_scores = [rouge.get_scores(hyp, ref) for hyp, ref in zip(hypotheses, references)]
    # Extract and average the ROUGE-L scores (or ROUGE-1, ROUGE-2 as needed)
    rouge_score = sum([score[0]['rouge-l']['f'] for score in rouge_scores]) / len(rouge_scores)
    rouge_score_1 = sum([score[0]['rouge-l']['p'] for score in rouge_scores]) / len(rouge_scores)
    rouge_score_2 = sum([score[0]['rouge-l']['r'] for score in rouge_scores]) / len(rouge_scores)
    print(rouge_score)
    print(rouge_score_1)
    print(rouge_score_2)



if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = get_dataset(config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = QuestionGenerationModel(config.model_name, config.d_model, device)
    
    train_model(model, train_dataloader, val_dataloader, device, config.num_epochs, config.lr)

    model = QuestionGenerationModel(config.model_name, config.d_model, device)
    model.load_state_dict(torch.load(config.model_save_name))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, model_max_length = 512)

    # Evaluate the model on the validation dataset
    evaluate(model, tokenizer, device, test_dataloader)