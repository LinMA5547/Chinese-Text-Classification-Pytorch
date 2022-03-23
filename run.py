# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import models.adversarial as adv
import utils
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--adv', default="", type=str, help='Adversarial method')
parser.add_argument('--eps', default=0.01, type=float, help='epsilon for adv')
parser.add_argument('--alpha', default=0.01, type=float, help='alpha for adv')
parser.add_argument('--delta_init', default="random", type=str, help='adv delta init type')
parser.add_argument('--attack_iters', default=7, type=int, help='attack_iters for PGD')



args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    config.epsilon, config.lower_limit, config.upper_limit = utils.get_epsilon(model.embedding.weight.detach(),args.eps)
    config.epsilon = config.epsilon.to(config.device)
    config.lower_limit = config.lower_limit.to(config.device)
    config.upper_limit = config.upper_limit.to(config.device)
    config.eps = args.eps
    config.alpha = args.alpha
    config.delta_init = args.delta_init
    config.adv = args.adv
    config.attack_iters = args.attack_iters
    if args.adv == "FSGM":
        model = adv.FSGM(model,config)
    elif args.adv == "PGD":
        model = adv.PGD(model,config)
    elif args.adv == "Free":
        model = adv.Free(model,config)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
