import os, sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
from tqdm import tqdm
from termcolor import colored
from scheduler import LinearWarmupCosineAnnealingLR
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from torch.utils.data.distributed import DistributedSampler
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from DataLoader import VideoQADataLoader, VideoQAValidationDataLoader
from utils import todevice
from validate import validate
import model.TempAligner as TempAligner

from utils import todevice
from config import cfg, cfg_from_file 
import clip
import random
import time 
from SemanticAligner import SemanticAligner
import wandb
import torch.nn.functional as F
from copy import deepcopy
from line_profiler import profile
import time

# Initialize wandb
wandb.init(project="tem-adapter-remove")
mseloss = nn.MSELoss()


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.99991, device=None): #0.999 originally
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay 
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
    # # original
    def _update(self, model, update_fn):
        # print("ema decay: ", self.decay)
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    @torch.no_grad()
    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

# validation after 1 epoch of training
@profile
def validate_from_train(cfg, tempaligner, semanticaligner, clip_model, valid_loader, device, write_preds=False): # data ->valid_loader
    tempaligner.eval()
    clip_model.eval()
    semanticaligner.eval()
    print('validating...')
    total_acc, count, total_loss = 0.0, 0, 0.0
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []
    # Initialize counters for each question type
    # q_type_correct = {'U': 0, 'A': 0, 'F': 0, 'R': 0, 'C': 0, 'I': 0}
    # q_type_total = {'U': 0, 'A': 0, 'F': 0, 'R': 0, 'C': 0, 'I': 0}
    with torch.no_grad():
        for i, batch in enumerate(valid_loader): 
            # tensor_parts = batch[:-1]  # All items except the last one
            # q_types = batch[-1]        # The last item is q_type
            # print(q_types)
            _, answers, ans_candidates, batch_clips_data, question = [todevice(x, device) for x in batch]
            if cfg.train.batch_size == 1:
                answers = answers.to(device, non_blocking=True)
            else:
                answers = answers.to(device, non_blocking=True).squeeze()

            # batch_clips_data = batch_clips_data.permute(0,1,3,2) ####

            batch_size = answers.size(0)
            feat_dim = batch_clips_data.shape[-1]
            num_ans = ans_candidates.shape[1]
            ans_candidates = ans_candidates.view(-1, 77)
            with torch.no_grad():
                answers_features = clip_model.encode_text( ans_candidates ).float()
                question_features = clip_model.encode_text(question.squeeze())
            # import ipdb; ipdb.set_trace()
            answers_features = semanticaligner(answers_features, batch_clips_data).float()
            # question_features = clip_model.encode_text( question.squeeze() ).float()
            video_appearance_feat = batch_clips_data.contiguous().view(batch_size, -1, feat_dim)
            answers_features = answers_features.view(batch_size, num_ans, -1) 
            question_features = question_features.to(device, non_blocking=True).unsqueeze(1)
            answers = answers.cuda().squeeze()
            batch_inputs = [answers,  answers_features, video_appearance_feat, question_features]
            logits, visual_embedding_decoder, video_appearance_feat_ori = tempaligner(*batch_inputs) 

            # logits = logits.to(device)
            #########################Added, hinge loss###########################################################################
            # logits = logits.view(batch_size, 4)

            batch_agg = np.concatenate(np.tile(np.arange(batch_size).reshape([batch_size, 1]),
                                               [1, 4])) * 4  # [0, 0, 0, 0, 0, 5, 5, 5, 5, 1, ...]
            answers_agg = tile(answers, 0, 4)
            # import ipdb; ipdb.set_trace()

            loss_ce = torch.max(torch.tensor(0.0, device=device), 
                             1.0 + logits - logits[answers_agg + torch.tensor(batch_agg, device=device)]) #torch.from_numpy(batch_agg).cuda()
            loss_ce = loss_ce.sum()
            recon_loss = mseloss(visual_embedding_decoder, video_appearance_feat_ori)
            loss = 0.01 * loss_ce + recon_loss
            total_loss += loss
            avg_loss = total_loss / (i + 1)
            ##########################################################################################################
            preds = torch.argmax(logits.view(batch_size, 4), dim=1)

            agreeings = (preds == answers)
            if write_preds:
                if cfg.dataset.question_type not in ['action', 'transition', 'count']:
                    preds = logits.argmax(1)
                if cfg.dataset.question_type in ['action', 'transition']:
                    answer_vocab = valid_loader.vocab['question_answer_idx_to_token']
                else:
                    answer_vocab = valid_loader.vocab['answer_idx_to_token']
                for predict in preds:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        all_preds.append(predict.item())
                    else:
                        all_preds.append(answer_vocab[predict.item()])
                for gt in answers:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        gts.append(gt.item())
                    else:
                        gts.append(answer_vocab[gt.item()])
                for id in video_ids:
                    v_ids.append(id.cpu().numpy())
                for ques_id in question_ids:
                    q_ids.append(ques_id)

            total_acc += agreeings.float().sum().item()
            count += answers.size(0)
        acc = total_acc / count
        print('train set size:',count)
        print('acc on trainset:',acc)

    if not write_preds:
        return acc, avg_loss
    else:
        return acc, all_preds, gts, v_ids, q_ids

# train code
@profile
def train(cfg, args):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'annotation_file': cfg.dataset.annotation_file,
        'annotation_file_qna': 'data/sutd-traffic/final_SUTD_qa_without_blank_allU.jsonl',
        'appearance_feat': cfg.dataset.appearance_feat,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'pin_memory': True,
        'shuffle': True,
        # 'drop_last': True
    }

    valid_loader_kwargs = {
        'annotation_file': './data/sutd-traffic/output_file_test.jsonl',
        'appearance_feat': cfg.dataset.appearance_feat,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'pin_memory': True,
        'shuffle': True,
    } 
    train_loader = VideoQADataLoader(**train_loader_kwargs)
    valid_loader = VideoQAValidationDataLoader(**valid_loader_kwargs) # Dataloader for validation
    logging.info("number of train instances: {}".format(len(train_loader.dataset)))
    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device) 
    clip_model.float()
    model_kwargs = {
        'vision_dim': cfg.train.vision_dim,
        'module_dim': cfg.train.module_dim,
    }
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}
    tempaligner = TempAligner.TempAligner(**model_kwargs).to(device)

    semanticaligner = SemanticAligner().to(device)
    optimizer = optim.AdamW(
    [
        {"params": tempaligner.parameters(), 'lr': cfg.train.lr},
        {"params": semanticaligner.parameters(), 'lr': cfg.train.lr},
    ]
    )

    warmup_epochs_set = 2
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs_set, max_epochs=cfg.train.max_epochs)

    mseloss = nn.MSELoss()
    start_epoch = 0
    best_val = 0
    tem_val_accuracy_ema=0.0
    valid_steps = 0
    semanticaligner_ema = ModelEma(semanticaligner, device=device)
    tempaligner_ema = ModelEma(tempaligner, device=device)
    if cfg.train.restore:
        print("Restore checkpoint and optimizer...")
        ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        start_epoch = ckpt['epoch'] + 1
        tempaligner.load_state_dict(ckpt['state_dict'])
        semanticaligner.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logging.info("Start training........")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.train.max_epochs):
        logging.info('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch), "green", attrs=["bold"])))
        tempaligner.train()
        semanticaligner.train()
        total_acc, count = 0, 0
        batch_mse_sum = 0.0
        total_loss, avg_loss = 0.0, 0.0
        avg_loss = 0
        total_ce_loss = 0.0
        avg_ce_loss = 0.0
        total_recon_loss = 0.0
        avg_recon_loss = 0.0
        train_accuracy = 0
        val_avg_loss, val_accuracy = 0.0, 0.0
        loss_dict = {}
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            progress = epoch + i / len(train_loader)
            ETA = (cfg.train.max_epochs - epoch) * (time.time() - start_time) / (i + 1)
            _, answers, ans_candidates, batch_clips_data, question = [todevice(x, device) for x in batch] 
            # batch_clips_data = batch_clips_data.permute(0,1,3,2) ####

            batch_size = batch_clips_data.shape[0]
            feat_dim = batch_clips_data.shape[-1]
            num_ans = ans_candidates.shape[1] 
            ans_candidates = ans_candidates.view(-1, 77)

            with torch.no_grad():
                answers_features = clip_model.encode_text( ans_candidates )
                question_features = clip_model.encode_text(question.squeeze())
            
            answers_features_before_semanticaligner = answers_features
            question_features_before_tempaligner = question_features
            answers_features = semanticaligner(answers_features, batch_clips_data)
            
            video_appearance_feat = batch_clips_data.contiguous().view(batch_size, -1, feat_dim) 
            answers_features_before_view = answers_features
            answers_features = answers_features.view(batch_size, num_ans, -1)
            answers = answers.to(device, non_blocking=True).squeeze()
            question_features = question_features.to(device, non_blocking=True).unsqueeze(1)
            # import ipdb; ipdb.set_trace()
            batch_inputs = [answers,  answers_features, video_appearance_feat, question_features]
            ################## Add model ema #########################
            # start_time_ema = time.time()
            
            # end_time_ema = time.time()
            # logging.info(f"EMA update took {end_time_ema - start_time_ema:.2f} seconds")
            ################## Add model ema #########################

            logits, visual_embedding_decoder, video_appearance_feat_ori = tempaligner(*batch_inputs) 
            batch_agg = np.concatenate(np.tile(np.arange(batch_size).reshape([batch_size, 1]),
                                               [1, 4])) * 4  # [0, 0, 0, 0, 0, 5, 5, 5, 5, 1, ...]
            answers_agg = tile(answers, 0, 4)
            # import ipdb; ipdb.set_trace()
            loss_ce = torch.max(torch.tensor(0.0, device=device),
                             1.0 + logits - logits[answers_agg + torch.from_numpy(batch_agg).to(device, non_blocking=True)])
            loss_ce = loss_ce.sum()

            recon_loss = mseloss(visual_embedding_decoder, video_appearance_feat_ori) 
            loss = 0.01*loss_ce + recon_loss
            loss.backward()
            total_loss += loss.detach()
            avg_loss = total_loss / (i + 1)
            total_ce_loss += loss_ce.detach()
            avg_ce_loss = total_ce_loss / (i+1)
            total_recon_loss += recon_loss.detach()
            avg_recon_loss = total_recon_loss / (i+1)
            nn.utils.clip_grad_norm_(tempaligner.parameters(), max_norm=12)
            # nn.utils.clip_grad_norm_(tempaligner.parameters(), max_norm=1)
            optimizer.step()
            semanticaligner_ema.update(semanticaligner)
            tempaligner_ema.update(tempaligner)
            preds = torch.argmax(logits.view(batch_size, 4), dim=1)
            aggreeings = (preds == answers)

            total_acc += aggreeings.sum().item()
            count += answers.size(0)
            train_accuracy = total_acc / count
            sys.stdout.write(
                "\rProgress = {progress} sum_loss = {sum_loss}  avg_loss = {avg_loss}  ce_loss = {ce_loss}  recon_loss = {recon_loss}  avg_acc = {avg_acc}  exp: {exp_name}  ETA: {ETA}m".format(
                    progress=colored("{:.1f}".format(progress), "green", attrs=['bold']),
                    sum_loss=colored("{:.3f}".format(loss.item()), "blue", attrs=['bold']),
                    avg_loss=colored("{:.3f}".format(avg_loss), "red", attrs=['bold']),
                    ce_loss=colored("{:.3f}".format(avg_ce_loss.item()), "red", attrs=['bold']),
                    recon_loss=colored("{:.3f}".format(avg_recon_loss.item()), "green", attrs=['bold']),
                    avg_acc=colored("{:.3f}".format(train_accuracy), "red", attrs=['bold']),
                    exp_name=cfg.exp_name,
                    ETA=colored("{:.1f}".format(ETA), "yellow", attrs=['bold'])))
            sys.stdout.flush()
            loss_dict["train/avg_loss"] = avg_loss

        sys.stdout.write("\n")
        if scheduler is not None:
            scheduler.step()
        else:
            if (epoch + 1) % 10 == 0:
                optimizer = step_decay(cfg, optimizer)
        sys.stdout.flush()
        logging.info("Epoch = %s   avg_loss = %.3f    avg_acc = %.3f" % (epoch, avg_loss, train_accuracy))
        end_time = time.time()
        logging.info(f"Epoch {epoch} took {end_time - start_time:.2f} seconds")
        start_time = end_time

        if (epoch+1)%10==0:
            ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            else:
                assert os.path.isdir(ckpt_dir)
            save_checkpoint(epoch, tempaligner, optimizer, model_kwargs_tosave, os.path.join( ckpt_dir, 'tempaligner_{}_uniform256_noMask.pt'.format(epoch) ) )
            save_checkpoint(epoch, semanticaligner, optimizer, None, os.path.join( ckpt_dir, 'semanticaligner_{}_uniform256_noMask.pt'.format(epoch) ) )
            sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % (ckpt_dir))
            sys.stdout.flush()

        sys.stdout.write("\n")
        if (epoch + 1) % 1 == 0:
            start_time_val = time.time()
            val_accuracy, val_avg_loss = validate_from_train(cfg, tempaligner, semanticaligner, clip_model, valid_loader, device, write_preds=False)
            end_time_val = time.time()
            logging.info(f"Val validation took {end_time_val - start_time_val:.2f} seconds")
            start_time_val_ema = time.time()
            val_accuracy_ema, val_avg_loss_ema = validate_from_train(cfg, tempaligner_ema.module, semanticaligner_ema.module, clip_model, valid_loader, device, write_preds=False)
            end_time_val_ema = time.time()
            logging.info(f"EMA validation took {end_time_val_ema - start_time_val_ema:.2f} seconds")
            if tem_val_accuracy_ema <= val_accuracy_ema or tem_val_accuracy <= val_accuracy:
                print("Epoch: ", epoch)
                print("Save the model with the highest validation accuracy with epoch, : ", tem_val_accuracy_ema, " -> ", val_accuracy_ema)
                ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                else:
                    assert os.path.isdir(ckpt_dir)
                
                # postfix = '_highestVal_EMA9997'
                # postfix = '_highestVal_EMA9998'
                # postfix = '_highestVal_EMA9999'
                postfix = '_highestVal_EMA99991'

                # model_name = 'GPT'
                model_name = 'T5'
                # model_name = 'None'
                task_name = 'AtoU'

                save_checkpoint(epoch, tempaligner, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, f'tempaligner_{epoch}_{postfix}_{model_name}_{task_name}_{val_accuracy}_{cfg.train.batch_size}_{warmup_epochs_set}_{cfg.train.max_epochs}_no_cross_attention_with_learnable.pt'))
                save_checkpoint(epoch, semanticaligner, optimizer, None, os.path.join(ckpt_dir, f'semanticaligner_{epoch}_{postfix}_{model_name}_{task_name}_{val_accuracy}_{cfg.train.batch_size}_{warmup_epochs_set}_{cfg.train.max_epochs}_no_cross_attention_with_learnable.pt'))    
                save_checkpoint(epoch, tempaligner_ema.module, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, f'tempaligner_ema_{epoch}_{postfix}_{model_name}_{task_name}_{val_accuracy_ema}_{cfg.train.batch_size}_{warmup_epochs_set}_{cfg.train.max_epochs}_no_cross_attention_with_learnable.pt'))# 여기도 val_accuracy값으로 이름이 저장되고 있었음
                save_checkpoint(epoch, semanticaligner_ema.module, optimizer, None, os.path.join(ckpt_dir, f'semanticaligner_ema_{epoch}_{postfix}_{model_name}_{task_name}_{val_accuracy_ema}_{cfg.train.batch_size}_{warmup_epochs_set}_{cfg.train.max_epochs}_no_cross_attention_with_learnable.pt'))
                tem_val_accuracy_ema = val_accuracy_ema
                tem_val_accuracy = val_accuracy
            tempaligner.train()
            semanticaligner.train()

            loss_dict["eval/val_avg_loss"] = val_avg_loss
            loss_dict["eval/val_accuracy"] = val_accuracy
            loss_dict["eval/val_avg_loss_ema"] = val_avg_loss_ema
            loss_dict["eval/val_accuracy_ema"] = val_accuracy_ema
        sys.stdout.flush()
        # logging.info("Epoch = %s   avg_loss = %.3f    avg_acc = %.3f" % (epoch, avg_loss, train_accuracy))
        logging.info("Epoch = %s    train_avg_loss = %.3f    val_avg_loss = %.3f    val_accuracy = %.3f" % (epoch, avg_loss, val_avg_loss, val_accuracy))
        wandb.log(loss_dict)

# Credit https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


# Replace this function with a pure PyTorch implementation
# def tile(a, dim, n_tile):
#     init_dim = a.size(dim)
#     repeat_idx = [1] * a.dim()
#     repeat_idx[dim] = n_tile
#     a = a.repeat(*(repeat_idx))
    
#     # Replace NumPy with PyTorch operations
#     idx_list = [init_dim * i + j for i in range(n_tile) for j in range(init_dim)]
#     order_index = torch.tensor(idx_list, device=a.device, dtype=torch.long)
#     return torch.index_select(a, dim, order_index)


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='sutd-traffic_transition.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    assert cfg.dataset.name in ['sutd-traffic']

    if not cfg.multi_gpus:
        torch.cuda.set_device(cfg.gpu_id)
    # make logging.info display into both shell and file
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    else:
        assert os.path.isdir(cfg.dataset.save_dir)
    log_file = os.path.join(cfg.dataset.save_dir, "log")
    if not cfg.train.restore and not os.path.exists(log_file):
        os.mkdir(log_file)
    else:
        assert os.path.isdir(log_file)

    fileHandler = logging.FileHandler(os.path.join(log_file, 'stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))
    # concat absolute path of input files

    if cfg.dataset.name == 'sutd-traffic':
        cfg.dataset.annotation_file = cfg.dataset.annotation_file.format('train')
        cfg.dataset.annotation_file_val = cfg.dataset.annotation_file.format('test')
        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))

    else:
        pass

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train(cfg, args)


if __name__ == '__main__':
    main()
