from transformers import AutoModel

model_name = 'facebook/bart-base'

temp_model = AutoModel.from_pretrained(model_name)
d_model = temp_model.config.d_model
del temp_model

batch_size = 8
num_epochs = 10
lr = 3e-6

model_save_name = 'QG_SQuAD_1_BART_base.pt'