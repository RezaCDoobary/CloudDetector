import torch
from tqdm import tqdm
from .util import get_metrics

class training(object):
    def __init__(self, 
                 loss_fn, 
                 optimiser, 
                 model,
                train_loader,
                val_loader):
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log = {}

    def load_data(self, data):
        X,y = data['data'], data['category']
        X = torch.transpose(X.float(),1,2)
        return X, y
    
    def perform_optimisation(self,X, y):
        self.optimiser.zero_grad()
        model_output = self.model(X)
        loss = self.loss_fn(model_output, y)
        loss.backward()
        self.optimiser.step() 
        return loss
    
    def reporting(self, batch_print, running_loss):
        last_loss = running_loss / batch_print # average loss per batch
        return last_loss

    def run_epoch(self, epoch_idx):
        running_loss = 0
        last_loss = 0
        epoch_index = 0
        tracker = []
        batch_print = 1
        last_loss = -999
        val_loss = -999
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i,x in pbar:
            pbar.set_description(f"BATCH TRAIN LOSS {round(last_loss,5)} - VAL LOSS {round(val_loss,5)}")
            # load data
            X,y = self.load_data(x)
            loss = self.perform_optimisation(X, y)
            running_loss += loss.item()
            if i % batch_print == 0 and i!=0:
                last_loss = self.reporting(batch_print, running_loss)
                running_loss = 0
        
                #validate
                all_true_output = []
                all_model_output = []
                for i,x in enumerate(self.val_loader):
                    # load data
                    X,y = self.load_data(x)
                    model_output = self.model(X)
                    all_true_output.append(y)
                    all_model_output.append(model_output)
        
                all_true = torch.concat(all_true_output)
                all_model = torch.concat(all_model_output)
                loss = self.loss_fn(all_model, all_true)
                val_loss = loss.item()
        
        
        #validate
        all_true_output = []
        all_model_output = []
        for i,x in enumerate(self.val_loader):
            # load data
            X,y = self.load_data(x)
            model_output = self.model(X)
            all_true_output.append(y)
            all_model_output.append(model_output)
        
        all_true = torch.concat(all_true_output)
        all_model = torch.concat(all_model_output)
        loss = self.loss_fn(all_model, all_true)
        classification_output = torch.argmax(torch.exp(all_model),axis = 1)
        results = get_metrics(all_true, classification_output)
        results['loss'] = loss.item()
        self.log[epoch_idx] = results
        print(results)

    def run(self, num_epoch):
        for i in range(num_epoch):
            self.run_epoch(epoch_idx = i)