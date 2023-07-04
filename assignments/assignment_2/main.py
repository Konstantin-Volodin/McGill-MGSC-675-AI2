#%%
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", ".*does not have many workers.*")


# DATASET
class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, context_size):        
        self.raw_data = pd.read_csv(csv_path)
        # plays_to_consider = ['macbeth']
        # self.raw_data = self.raw_data.query(f"Play in {plays_to_consider}")

        # Created Raw Text (some character cleaning)
        self.raw_text = {}
        for name, dfg in self.raw_data.groupby('Play'):
            res = " ".join(i for i in dfg.PlayerLine.tolist())
            res = res.lower()
            self.raw_text[name] = res

        # Get Vocabulary
        self.vocab = set()
        for key, value in self.raw_text.items():
            self.vocab = self.vocab.union(set(value))
        self.vocab_size = len(self.vocab)
        self.char_to_int = {char: i for i, char in enumerate(self.vocab)}

        # Get Context and Target from data
        self.context_size = context_size
        self.context = []
        self.target = []
        for key, value in self.raw_text.items():
            for i in range(self.context_size, len(value)):
                context_t = [self.char_to_int[value[i-j]] 
                             for j in range(self.context_size, 0, -1)] 
                target_t = self.char_to_int[value[i]]
                self.context.append(context_t)
                self.target.append(target_t)

        # Tensor Encoding
        self.context_emb = torch.tensor(self.context)
        self.target_emb = torch.tensor(self.target)

    def __len__(self):
        return self.context_emb.shape[0]

    def __getitem__(self, idx):
        input = self.context_emb[idx, :]
        output = self.target_emb[idx]
        # input_tensor = torch.tensor(input)
        # output_tensor = torch.tensor(output)
        return input, output

# MODEL
class TextGenerator(pl.LightningModule):
    def __init__(self, config, info):
        super(TextGenerator, self).__init__()
        self.num_layers = config['NUM_LAYERS']
        self.emb = config['EMBEDDING']
        self.dropout = config['DROPOUT']
        self.hidden_size = config['HIDDEN_DIM']
        self.epochs = config['MAX_EPOCHS']
        self.batch_size = info['BATCH_SIZE']
        self.text_context = info['TEXT_CONTEXT']
        self.vocab = info['VOCAB_SIZE']

        self.embedding = nn.Embedding(num_embeddings=self.vocab,
                                      embedding_dim=self.emb)
        self.lstm = nn.LSTM(input_size = self.emb, 
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers,
                            dropout=self.dropout,
                            batch_first=True)
        self.out = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.Linear(self.hidden_size, self.vocab),)
        
        # MODEL 
        self.model = nn.Sequential(self.lstm,
                                   self.out)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.epochs)
        return [optim], [scheduler]
        

    def forward(self, x, hidden=None):
        embed = self.embedding(x)

        if hidden == None:
            output, hidden = self.lstm(embed)
        else:
            output, hidden = self.lstm(embed, hidden)
        
        preds = self.out(output[:,-1,:])

        hidden = (hidden[0].detach(),
                  hidden[1].detach(),)

        return preds, hidden
    
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if batch_idx == 0: y_pred, self.hidden = self.forward(x)
        else: y_pred, self.hidden = self.forward(x, self.hidden)

        loss = nn.functional.cross_entropy(y_pred, y)
        acc = sum(y_pred.argmax(dim=1) == y)/self.batch_size
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if batch_idx == 0: y_pred, self.hidden = self.forward(x)
        else: y_pred, self.hidden = self.forward(x, self.hidden)

        loss = nn.functional.cross_entropy(y_pred, y)
        acc = sum(y_pred.argmax(dim=1) == y)/self.batch_size
        self.log('valid_loss', loss, prog_bar=True)
        self.log('valid_acc', acc, prog_bar=True, on_epoch=True)
        return loss


#%% LOAD DATA
info = {'BATCH_SIZE': 64,
        'TEXT_CONTEXT': 50}
data = ShakespeareDataset('data/data.csv', info['TEXT_CONTEXT'])

#%%
data_train, data_val, data_test, _ = torch.utils.data.random_split(
    data, [32*10000, 32*50, 32*50, data.__len__() - (32*10000) - 32*50 - 32*50]
    # data, [128*100, 128*5, 128*5, data.__len__()- (128*100) - (128*5) - (128*5)]
)

dltrain = DataLoader(data_train, batch_size=info['BATCH_SIZE'], shuffle=True)
dlval = DataLoader(data_val, batch_size=info['BATCH_SIZE'], shuffle=False)
dltest = DataLoader(data_test, batch_size=info['BATCH_SIZE'], shuffle=False)

hyper = {'HIDDEN_DIM': 256,
         'EMBEDDING': 32,
         'NUM_LAYERS': 4,
         'DROPOUT': 0.5,
         'MAX_EPOCHS': 100}
info['VOCAB_SIZE'] = data.vocab_size

# MODEL
model = TextGenerator(hyper, info)
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='valid_loss', save_top_k=-1, mode='min')
trainer = pl.Trainer(max_epochs=hyper['MAX_EPOCHS'],callbacks=[checkpoint_callback]) 
trainer.fit(model, dltrain, dlval)


# %% PREDICTION
input_sent = "the luminous moon gr"
print(len(input_sent))

int_to_char = {v: k for k, v in data.char_to_int.items()}
input_sent_prep = [data.char_to_int[i] for i in input_sent]

prediction_length = 20
pred_sentence = []

for i in range(prediction_length):
    if i != 0: del input_sent_prep[0]    
    
    input_tensor = torch.tensor(input_sent_prep).reshape(1,20)
    res, _ = model.forward(input_tensor)
    res = int(torch.argmax(res))

    pred_sentence.append(res)
    input_sent_prep.append(res)

pred_sentence_clean = "".join([int_to_char[i] for i in pred_sentence])
"".join([input_sent,pred_sentence_clean])

# %%
