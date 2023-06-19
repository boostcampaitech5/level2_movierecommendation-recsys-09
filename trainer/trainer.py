import numpy as np
import os
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, EarlyStopping, Recall_at_k_batch
import sys
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import wandb
from time import time
from model.metric import ndcg_k, recall_at_k

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    
###########################################################################################################################    
class Trainer_new:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        self.args = args['trainer']
        #self.cuda_condition = torch.cuda.is_available() and not args["no_cuda"]
        self.cuda_condition = True
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args["adam_beta1"], self.args["adam_beta2"])
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args["lr"],
            betas=betas,
            weight_decay=self.args["weight_decay"],
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "RECALL@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "RECALL@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args["hidden_size"])  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.model.args["max_seq_length"]).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    

class AutoRecTrainer(Trainer_new):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(AutoRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )
        self.train_matrix = np.load('/opt/ml/input/data/train_mat.npy')
        self.wandb = args["wandb"]
        self.epochs = args["trainer"]["epochs"]
        self.wandb_name = args["wandb_name"]
        self.patience = args["patience"]
        self.output_dir = args['output_dir']
        
        self.args = args['lr_scheduler']
        self.loss_fn = nn.MSELoss().to(self.device)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            'min',
            factor = self.args["scheduler_factor"],
            eps = self.args["scheduler_eps"],
            patience = self.args["scheduler_patience"],
            )

    def iteration(self, epoch, dataloader, mode="train"):

        # Setting the tqdm progress bar

        rec_data_iter = tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )
        if mode == "train":
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, inter_mat, _ = batch

                pred = self.model(inter_mat)
                loss = self.loss_fn(pred, inter_mat)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()


            rec_avg_loss /= len(rec_data_iter)

            self.scheduler.step(rec_avg_loss)

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss),
                "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
            }

            if (epoch + 1) % 1 == 0:
                print(str(post_fix))

        else:
            self.model.eval()

            pred_list = None
            answer_list = None
            with torch.no_grad():
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, inter_mat, answers = batch

                    rating_pred = self.model(inter_mat)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.train_matrix[batch_user_index] > 0] = -1

                    ind = np.argpartition(rating_pred, -10)[:, -10:]
                    # ind = np.argpartition(rating_pred, -20)[:, -20:]

                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                    batch_pred_list = ind[
                        np.arange(len(rating_pred))[:, None], arr_ind_argsort
                    ]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(
                            answer_list, answers.cpu().data.numpy(), axis=0
                        )

            if mode == "submission":
                return pred_list
            else:
                return self.get_full_sort_score(epoch, answer_list, pred_list)
            
    def train_and_validate(self):
        checkpoint = self.wandb_name + ".pt"
        self.early_stopping = EarlyStopping(os.path.join(self.output_dir, checkpoint), patience=self.patience, verbose=True)
        for epoch in tqdm(range(self.epochs)):
            self.train(epoch)

            scores, _ = self.valid(epoch)
            # wandb log
            if self.wandb:
                wandb.log({
                    "recall@5": scores[0],
                    "ndcg@5": scores[1],
                    "recall@10": scores[2],
                    "ndcg@10": scores[3]
                })

            self.early_stopping(np.array([scores[2]]), self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

class Trainer_ML():
    def __init__(self, model, config, data_loader, valid_data_loader):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

    def train(self):
        self.model.train(self.data_loader.train_dataset, self.valid_data_loader)
        self.model.save_model_pkl(self.config["trainer"]["save_dir"]+self.config["trainer"]["save_model_path"])

        
class MVAE_Trainer():
    def __init__(self, model, criterion, config, data_loader, valid_data_loader, optimizer):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.train_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.optimizer = optimizer
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

    def train(self):
        update_count = 0
        best_r10 = -np.inf

        for epoch in range(1, self.config['trainer']['epochs'] + 1):
            epoch_start_time = time()
            ###### train ######
            self.model.train()
            train_loss = 0.0
            start_time = time()

            for batch_idx, batch_data in enumerate(self.train_loader):
                input_data = batch_data.to(self.device)
                self.optimizer.zero_grad()
                if self.config['trainer']['total_anneal_steps'] > 0:
                    anneal = min(self.config['trainer']['anneal_cap'], 
                                    1. * update_count / self.config['trainer']['total_anneal_steps'])
                else:
                    anneal = self.config['trainer']['anneal_cap']

                recon_batch, mu, logvar = self.model(input_data)
                
                loss = self.criterion(recon_batch, input_data, mu, logvar, anneal)
                
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

                update_count += 1        

                log_interval = 100
                if batch_idx % log_interval == 0 and batch_idx > 0:
                    elapsed = time() - start_time
                    print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                            'loss {:4.2f}'.format(
                                epoch, batch_idx, len(range(0, 6807, self.config['data_loader']['args']['train_batch_size'])),
                                elapsed * 1000 / log_interval,
                                train_loss / log_interval))

                    start_time = time()
                    train_loss = 0.0

            ###### eval ######
            recall10_list = []
            recall20_list = []
            total_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch_data in self.valid_data_loader:
                    input_data, label_data = batch_data # label_data = validation set 추론에도 사용되지 않고 오로지 평가의 정답지로 사용된다. 
                    input_data = input_data.to(self.device)
                    label_data = label_data.to(self.device)
                    label_data = label_data.cpu().numpy()
                    
                    if self.config['trainer']['total_anneal_steps'] > 0:
                        anneal = min(self.config['trainer']['anneal_cap'], 
                                    1. * update_count / self.config['trainer']['total_anneal_steps'])
                    else:
                        anneal = self.config['trainer']['anneal_cap']

                    recon_batch, mu, logvar = self.model(input_data)

                    loss = self.criterion(recon_batch, input_data, mu, logvar, anneal)

                    total_loss += loss.item()
                    recon_batch = recon_batch.cpu().numpy()
                    recon_batch[input_data.cpu().numpy().nonzero()] = -np.inf

                    recall10 = Recall_at_k_batch(recon_batch, label_data, 10)
                    recall20 = Recall_at_k_batch(recon_batch, label_data, 20)
                    
                    recall10_list.append(recall10)
                    recall20_list.append(recall20)
            
            total_loss /= len(range(0, 6807, 1000))
            r10_list = np.concatenate(recall10_list)
            r20_list = np.concatenate(recall20_list)
                    
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                    'r10 {:5.3f} | r20 {:5.3f}'.format(
                        epoch, time() - epoch_start_time, total_loss, np.mean(r10_list), np.mean(r20_list)))
            print('-' * 89)
            
            if(self.config['wandb']):
                wandb.log({"valid loss" : total_loss,
                "r20" : np.mean(r20_list), 
                "r10" : np.mean(r10_list)})

            if np.mean(r10_list) > best_r10:
                with open(self.config['test']['save'], 'wb') as f:
                    torch.save(self.model, f)
                best_r10 = np.mean(r10_list)
