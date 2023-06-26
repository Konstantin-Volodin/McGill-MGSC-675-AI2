#%%
import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

class CIFARClassifier(pl.LightningModule):
    def __init__(self, model, epochs_max):
        super().__init__()
        self.lr = 0.01
        self.em = epochs_max

        if model == 'resnet34':
            self.model = torchvision.models.resnet34(torchvision.models.ResNet34_Weights)
            self.model.fc = nn.Linear(512, 10)

        if model == 'resnet18':
            self.model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights)
            self.model.fc = nn.Linear(512, 10)

        elif model == 'effnet_s':            
            self.model = torchvision.models.efficientnet_v2_s(torchvision.models.EfficientNet_V2_S_Weights)
            self.model.classifier[-1] = nn.Linear(1280, 10)

        elif model == 'effnet_m':            
            self.model = torchvision.models.efficientnet_v2_m(torchvision.models.EfficientNet_V2_M_Weights)
            self.model.classifier[-1] = nn.Linear(1280, 10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).sum().item() / y.size(0)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.em)
        return [optimizer], [scheduler]



#%% DATA PREP
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=3),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                        std=[0.229, 0.224, 0.225])])

batch_size = 128
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,)


#%% EVALUATION OF VARIOUS METHODS
models_to_try = ['resnet18','resnet34', 'effnet_s', 'effnet_m']
for model in models_to_try:
    model = CIFARClassifier(model, 20)
    trainer = pl.Trainer(max_epochs=20, log_every_n_steps=5)
    trainer.fit(model, train_loader, val_loader)


# %% FINAL MODEL
model = CIFARClassifier('effnet_s', 250)
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=-1, mode='min')
trainer = pl.Trainer(max_epochs=250, log_every_n_steps=5, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)
