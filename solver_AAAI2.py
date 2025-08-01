# Imports
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD, MMD_loss, get_shuffled,Criterion, loss_fn,qlog,apply_qlog_to_tensor,qexp,quaternion_angular_error
import model_AAAI2

from sklearn.metrics import mean_squared_error 
import os.path as osp
from torchvision import transforms, models


import os

class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):
        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
        self.train_criterion = Criterion(sax=self.train_config.sax,saq=self.train_config.saq, learn_beta=True)

    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(model_AAAI2, self.train_config.model)(self.train_config)
        # Final list
        for name, param in self.model.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            print('\t' + name, param.requires_grad)


        self.param_list = [{'params': self.model.parameters()}]
        if hasattr(self.train_config, 'sax') and hasattr(self.train_config, 'saq'):
            print('learn_beta')
            self.param_list.append({'params': [self.train_criterion.sax, self.train_criterion.saq]})


        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                self.param_list,
                #filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)
    def set_all_seeds(self,seed=42):

        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @time_desc_decorator('Training Start!')
    def train(self):
        self.name="Sub5-temporal-1"
        self.criterion1=criterion=self.train_criterion
        self.set_all_seeds(42)
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_sim = nn.TripletMarginLoss(margin=1.0, p=2.0) #nn.MSELoss() #MMD_loss() #CMD()
        self.L=int(500//3)
        train_losses = []

        eeg_best_tloss = float(50)  
        eeg_best_qloss = float('inf')  
        image_best_tloss = float(50)  
        image_best_qloss = float('inf')  
        e1,e2,e3,e4=0,0,0,0

        for e in range(self.train_config.n_epoch):
            self.model=to_gpu(self.model)
            self.model.train()
            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            train_loss = []
            
            for batch in self.train_data_loader:
                self.optimizer.zero_grad()
                self.model.zero_grad()
                eeg, label, image = batch
                #---------------------------------------占位符设计----------------------
                # p = 0.1
                # B = image.size(0)
                # random_selector = torch.rand(B)  # 每个样本生成一个 [0, 1) 的随机数

                # for i in range(B):
                #     if random_selector[i] < p:
                #         image[i] = torch.zeros_like(image[i])
                # torch.manual_seed(42)  

                B = image.size(0)
                random_selector = torch.randint(0, 2, (B,))  # 0 或 1
                for i in range(B):
                    if random_selector[i] == 1:
                        image[i] = torch.zeros_like(image[i])
                #print(image)
                #---------------------------------------占位符设计----------------------
                y=np.zeros((len(label), 6))
                for i in range(len(y)):
                    p = label[i, :3]  
                    q = label[i, 3:7]
                    q *= np.sign(q[0])  # constrain to hemisphere
                    back=qlog(q)
                    y[i] = np.hstack((p, back))
                y=torch.tensor(y)
                eeg = to_gpu(eeg)
                image = to_gpu(image)
                label = to_gpu(y)
                
                y_tilde_eeg, y_tilde_image = self.model(eeg, image)

                
                #label=to_gpu(label1)
                y_tilde_eeg=to_gpu(y_tilde_eeg)
                y_tilde_image=to_gpu(y_tilde_image)

                with torch.no_grad():
                    placeholder_mask = (image.view(B, -1).abs().sum(dim=1) == 0)  # [B], bool，True表示是占位图像

                loss_eeg = criterion(y_tilde_eeg, label)
                loss_image = criterion(y_tilde_image, label)
                
                image_loss_weights = torch.where(placeholder_mask, 1.0, 1.0)
                loss_image = (loss_image.view(-1) * image_loss_weights).mean()


                cls_loss = loss_eeg + loss_image
                loss = cls_loss * self.train_config.cla_weight 
                loss.backward()
                self.optimizer.step()
                train_loss_cls.append(cls_loss.item())
                train_loss.append(loss.item())

            train_losses.append(train_loss)
            print("Epoch {0}: loss={1:.4f}".format(e, np.mean(train_loss)))

                     
            if e % 10 == 0:  # 每10个epoch
                
                _,_ ,eeg_tloss, eeg_tvariance, eeg_tstd, eeg_qloss, eeg_qvariance, eeg_qstd,_,_= self.eval_E2I(mode="dev")

                print("当前最佳 EEG tloss: {:.6f} 在 epoch:{:.1f}, qloss: {:.6f} 在 epoch:{:.1f}".format(eeg_best_tloss, e1, eeg_best_qloss, e2))

                print("当前 EEG tloss 标准差: {:.4f}, 方差: {:.4f}".format(eeg_tstd, eeg_tvariance))
                print("当前 EEG qloss 标准差: {:.4f}, 方差: {:.4f}".format(eeg_qstd, eeg_qvariance))

                #test
                with open(f' /test/BCML/SAVE_AAAI/new_model(token)/{self.name}.txt', 'a') as log_file:
                    log_file.write("Epoch {}: 当前最佳 EEG tloss: {:.6f}, qloss: {:.6f}\n".format(
                        e, eeg_best_tloss, eeg_best_qloss))
                    log_file.write("Epoch {}: EEG tloss: {:.6f}, EEG qloss: {:.6f}\n".format(
                        e, eeg_tloss, eeg_qloss))
                    log_file.write("Epoch {}: EEG tloss 标准差: {:.4f}, 方差: {:.4f}, EEG qloss 标准差: {:.4f}, 方差: {:.4f}\n".format(
                        e, eeg_tstd, eeg_tvariance, eeg_qstd, eeg_qvariance))
            else:
                _,_ ,eeg_tloss, eeg_tvariance, eeg_tstd, eeg_qloss, eeg_qvariance, eeg_qstd,_,_= self.eval_E2I(mode="dev")
                with open(f' /test/BCML/SAVE_AAAI/new_model(token)/{self.name}.txt', 'a') as log_file:
                    log_file.write("Epoch {}: EEG tloss: {:.6f}, EEG qloss: {:.6f}\n".format(
                        e, eeg_tloss, eeg_qloss))
                    log_file.write("Epoch {}: EEG tloss 标准差: {:.4f}, 方差: {:.4f}, EEG qloss 标准差: {:.4f}, 方差: {:.4f}\n".format(
                        e, eeg_tstd, eeg_tvariance, eeg_qstd, eeg_qvariance))
            if eeg_tloss < eeg_best_tloss:
                eeg_best_tloss = eeg_tloss 
                e1 = e 
                _, _, _, _,_,_,_,_, predeg_poses,targ_poses= self.eval_E2I(mode="dev")      
                model_dir = ' /test/BCML/models_AAAI/new_model(Token)'
                save_dir = ' /test/BCML/memory_AAAI/new_model(token)'
                os.makedirs(save_dir, exist_ok=True)

                # 构建完整的文件路径
                pred_path = os.path.join(save_dir, f'{self.name}_predicted_poses.txt')
                targ_path = os.path.join(save_dir, f'{self.name}_target_poses.txt')
                best_model_params = {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        'criterion_params': {
                            'sax': self.train_criterion.sax.data.clone(),
                            'saq': self.train_criterion.saq.data.clone(),
                        }
                    }
                    
                best_model_path = os.path.join(model_dir, f"{self.name}_best.pth")
                torch.save(best_model_params, best_model_path)
                
                # print(f"最佳模型 - Epoch {e}, Loss: {eeg_best_tloss:.6f}")
                # print(f"预测结果已保存至: {pred_path}")
                # print(f"目标结果已保存至: {targ_path}")
                # print(f"最佳模型参数已保存至: {best_model_path}")
                np.savetxt(pred_path, predeg_poses, fmt='%.6f')
                np.savetxt(targ_path, targ_poses, fmt='%.6f')
                
                
                
            if eeg_qloss < eeg_best_qloss:  
                eeg_best_qloss = eeg_qloss
                e2 = e  

        
        
    def eval_E2I(self,mode=None, to_print=None,plot=None):
        self.model=self.model.to("cpu")
        eval_model=self.model
        eval_model.image_model.droprate = 0
        eval_model==to_gpu(eval_model)
        assert(mode is not None)
        eval_model.eval()
        y_true, y_pred_eeg, y_pred_image = [], [], []
        eval_loss_eeg, eval_loss_image = [], []
        L = self.L
        predeg_poses = np.zeros((L, 7))  # store all predicted poses
        predim_poses = np.zeros((L, 7))
        targ_poses = np.zeros((L, 7))  # store all target poses 
        pose_stats_file = osp.join(' /test/BCML/data/Work/pose_stats.txt')
        pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev
        t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        q_criterion = quaternion_angular_error
        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "train":
            dataloader = self.train_data_loader
        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                eeg, label, image = batch


                y=np.zeros((len(label), 6))
                for i in range(len(y)):
                    p = label[i, :3]  
                    q = label[i, 3:7]
                    q *= np.sign(q[0])  # constrain to hemisphere
                    back=qlog(q)
                    y[i] = np.hstack((p, back))  
                y=torch.tensor(y)
                eeg = to_gpu(eeg)
                image = to_gpu(image)
                y = to_gpu(y)
                #y_tilde_eeg = eval_model(eeg, image)
                
                y_tilde_eeg, _ = self.model(eeg, image)

                cls_loss_eeg = self.criterion1(y_tilde_eeg, y)
                
                loss_eeg = cls_loss_eeg





                eval_loss_eeg.append(loss_eeg.item())
                

                y_pred_eeg.append(y_tilde_eeg.detach().cpu().numpy())

                y_true.append(y.detach().cpu().numpy())        
                #eval pose  
                if len(y_true)==L:
                    y_true = np.concatenate(y_true, axis=0).squeeze()
                    y_pred_eeg = np.concatenate(y_pred_eeg, axis=0).squeeze()

                    p,q=[],[]
                    for i in range(len(y_true)):
                        s = torch.Size([1, 6])
                        eeg_output = y_pred_eeg[i].reshape((-1, s[-1]))
             
                        target = y_true[i].reshape((-1, s[-1]))
                        q = [qexp(p[3:]) for p in eeg_output]
                        eeg_output = np.hstack((eeg_output[:, :3], np.asarray(q)))

                        q = [qexp(p[3:]) for p in target]
                        target = np.hstack((target[:, :3], np.asarray(q)))
                        
                        eeg_output[:, :3] = (eeg_output[:, :3] * pose_s) + pose_m

                        target[:, :3] = (target[:, :3] * pose_s) + pose_m
                    # take the middle prediction
                        predeg_poses[i, :] = eeg_output[len(eeg_output) // 2]

                        targ_poses[i, :] = target[len(target) // 2]
        eeg_t_loss = np.asarray([t_criterion(p, t) for p, t in zip(predeg_poses[:, :3], targ_poses[:, :3])])
        eeg_q_loss = np.asarray([q_criterion(p, t) for p, t in zip(predeg_poses[:, 3:], targ_poses[:, 3:])])

        eval_loss_eeg = np.mean(eval_loss_eeg)
        eval_loss_image = np.mean(eval_loss_image)

        eeg_t_mean = np.mean(eeg_t_loss)
        eeg_t_variance = np.var(eeg_t_loss)
        eeg_t_std = np.std(eeg_t_loss)
        
        eeg_q_mean = np.mean(eeg_q_loss)
        eeg_q_variance = np.var(eeg_q_loss)
        eeg_q_std = np.std(eeg_q_loss)

        return (
            eval_loss_eeg,
            eval_loss_image,
            eeg_t_mean, eeg_t_variance, eeg_t_std,
            eeg_q_mean, eeg_q_variance, eeg_q_std,

            predeg_poses,
            targ_poses
        )


