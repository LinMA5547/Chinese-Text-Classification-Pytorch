import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

def clamp(X, lower_limit, upper_limit):
    # print (X.size(),lower_limit.size(),upper_limit.size())
    return torch.max(torch.min(X, upper_limit), lower_limit)
class Adversarial(nn.Module):
    def __init__(self, model):
        super(Adversarial, self).__init__()
        self.model = model
        self.disable_all_dropout()

    def disable_all_dropout(self):
        def disable_dropout(module):
            if isinstance(module, (nn.Dropout)):
                module.p = 0
        self.model.apply(disable_dropout)

    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)


class FSGM(Adversarial):
    def __init__(self, model, config, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        super(FSGM, self).__init__(model)
        self.emb_name = emb_name
        self.config = config
        self.backup = {}
        self.delta = nn.Embedding(config.n_vocab, config.embed)
        self.delta = self.delta.to(config.device)

    def forward(self, X,y):
        if self.training:
            # 正常梯度 + 对抗梯度
            self.config.epsilon, self.config.lower_limit, self.config.upper_limit = utils.get_epsilon(self.model.embedding.weight.detach(),self.config.eps)
            self.config.epsilon = self.config.epsilon.to(self.config.device)
            self.config.lower_limit = self.config.lower_limit.to(self.config.device)
            self.config.upper_limit = self.config.upper_limit.to(self.config.device)
            self.attack(X,y)  # 叠加对抗扰动
            output_adv = self.model(X)
            self.restore()  # 恢复embedding参数
            return output_adv
        else:
            return self.model(X)

    def attack(self,X, y):
        config = self.config
        delta_out = self.delta(X[0])
        delta_p = None
        for p in self.delta.parameters():
            delta_p = p
            break
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()

                if config.delta_init != 'previous':
                    delta_p.data = torch.zeros_like(param).to(config.device)
                if config.delta_init == 'random':
                    for i in range(self.delta.weight.data.size()[-1]):
                        delta_p.data[:,i].uniform_(-config.epsilon[0][i].item(),config.epsilon[0][i].item())
                    delta_p.data = clamp(delta_p.data, config.lower_limit - param.data, config.upper_limit - param.data)

                output = self.model(X,delta_out)
                loss = F.cross_entropy(output,y)
                loss.backward()
                grad = delta_p.grad.detach()
                delta_p.data = clamp(delta_p.data + config.alpha * torch.sign(grad), -config.epsilon, config.epsilon)
                delta_p.data = clamp(delta_p.data, config.lower_limit - param.data, config.upper_limit -param.data)
                delta_p = delta_p.detach()
                param.data.add_(delta_p.data)
        assert len(self.backup) >   0

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


class PGD(Adversarial):
    def __init__(self, model, config, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        super(PGD, self).__init__(model)
        self.emb_name = emb_name
        self.config = config
        self.backup = {}
        self.delta = nn.Embedding(config.n_vocab, config.embed)
        self.delta = self.delta.to(config.device)

    def forward(self, X,y):
        if self.training:
            # 正常梯度 + 对抗梯度
            self.config.epsilon, self.config.lower_limit, self.config.upper_limit = utils.get_epsilon(self.model.embedding.weight.detach(),self.config.eps)
            self.config.epsilon = self.config.epsilon.to(self.config.device)
            self.config.lower_limit = self.config.lower_limit.to(self.config.device)
            self.config.upper_limit = self.config.upper_limit.to(self.config.device)
            self.attack(X,y)  # 叠加对抗扰动
            output_adv = self.model(X)
            self.restore()  # 恢复embedding参数
            return output_adv
        else:
            return self.model(X)

    def attack(self,X, y):
        config = self.config
        
        delta_p = None
        for p in self.delta.parameters():
            delta_p = p
            break
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()

                # if config.delta_init != 'previous':
                delta_p.data = torch.zeros_like(param).to(config.device)
                if config.delta_init == 'random':
                    for i in range(self.delta.weight.data.size()[-1]):
                        delta_p.data[:,i].uniform_(-config.epsilon[0][i].item(),config.epsilon[0][i].item())
                    delta_p.data = clamp(delta_p.data, config.lower_limit - param.data, config.upper_limit - param.data)
                
                for _ in range(self.config.attack_iters):
                    delta_out = self.delta(X[0])
                    output = self.model(X,delta_out)
                    loss = F.cross_entropy(output,y)
                    loss.backward()
                    grad = delta_p.grad.detach()
                    delta_p.data = clamp(delta_p.data + (config.alpha / (self.config.attack_iters + 1)) * torch.sign(grad), -config.epsilon, config.epsilon)
                    delta_p.data = clamp(delta_p.data, config.lower_limit - param.data, config.upper_limit -param.data)
                    delta_p.grad.zero_()
                delta_p = delta_p.detach()
                param.data.add_(delta_p.data)
        assert len(self.backup) >   0

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}





class Free(Adversarial):
    def __init__(self, model, config, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        super(Free, self).__init__(model)
        self.emb_name = emb_name
        self.config = config
        self.backup = {}
        self.delta = nn.Embedding(config.n_vocab, config.embed)
        self.delta = self.delta.to(config.device)
        self.opt = None
        self.delta_p = None
        for p in self.delta.parameters():
            self.delta_p = p
            break
        self.delta_p.data = torch.zeros_like(self.delta_p.data).to(config.device)

    def forward(self, X,y):
        if self.training:
            # 正常梯度 + 对抗梯度
 
            loss,output = self.attack(X,y)  # 叠加对抗扰动
            # output_adv = self.model(X)
            # self.restore()  # 恢复embedding参数
            return loss,output
        else:
            return self.model(X)

    def attack(self,X, y):
        

        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                # self.backup[name] = param.data.clone()

                
                for _ in range(self.config.attack_iters):
                    self.config.epsilon, self.config.lower_limit, self.config.upper_limit = utils.get_epsilon(self.model.embedding.weight.detach(),self.config.eps)
                    self.config.epsilon = self.config.epsilon.to(self.config.device)
                    self.config.lower_limit = self.config.lower_limit.to(self.config.device)
                    self.config.upper_limit = self.config.upper_limit.to(self.config.device)
                    delta_out = self.delta(X[0])
                    output = self.model(X,delta_out)
                    loss = F.cross_entropy(output,y)
                    self.model.zero_grad()
                    loss.backward()
                    grad = self.delta_p.grad.detach()
                    self.delta_p.data = clamp(self.delta_p.data + (self.config.alpha / (0 + 1))  * torch.sign(grad), -self.config.epsilon, self.config.epsilon)
                    self.delta_p.data = clamp(self.delta_p.data, self.config.lower_limit - param.data, self.config.upper_limit -param.data)
                    self.opt.step()
                    self.delta_p.grad.zero_()
                return loss,output
                # delta_p = delta_p.detach()
                # param.data.add_(delta_p.data)
        # assert len(self.backup) >   0

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


