import argparse
import os
import time
import numpy as np
import torch
from Dataset import TrainingDataset, VTDataset, data_load
from Model_routing_21NA import Net as NA_NET
from Model_routing_21sp import Net
from torch.utils.data import DataLoader
from Train import train
from Full_t import full_t
from Full_vt import full_vt
# from torch.utils.tensorboard import SummaryWriter

from utils_log import Logger
from datetime import datetime
###############################248###########################################

def edgeNA(num_user, num_item, edge_index):
    edge_index = torch.tensor(edge_index).t().contiguous().cuda()
    edge,i = torch.sort(edge_index,-1)
    edge[0] = edge[0][i[-1]]
    m2 = torch.arange(num_item + num_user, num_item + num_user + num_item)
    m3 = torch.arange(num_item + num_user + num_item, num_item + num_user + num_item + num_item)
    value = edge[1][0]
    sind = 0
    count = 0
    new= []
    for index,v in enumerate(edge[1]):
        if v != value:
            count = 0
            value = v
            sind +=1
        if count == 0:
            new.append(torch.Tensor((v,m2[sind])))
            new.append(torch.Tensor((v,m3[sind])))
            new.append(torch.Tensor((m2[sind],m3[sind])))
        new.append(torch.Tensor((edge[0][index],m2[sind])))
        new.append(torch.Tensor((edge[0][index],m3[sind])))
        count += 1
    newt = torch.stack(new,-1).long().cuda()
    mm_edge = torch.cat([edge_index,newt],-1)
        
    return mm_edge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    parser.add_argument('--save_file', default='', help='Filename')

    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')

    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--alpha_threshold', type=str, default='>0.5', help='alpha threshold.')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Validation Batch size.')

    parser.add_argument('--num_epoch', type=int, default=280, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument('--num_routing', type=int, default=3, help='Layer number.')

    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--dim_C', type=int, default=64, help='Latent dimension.')

    parser.add_argument('--dropout', type=float, default=0, help='dropout.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation Mode.')
    parser.add_argument('--topK', type=int, default=10, help='Workers number.')

    parser.add_argument('--has_act', default='False', help='Has non-linear function.')
    parser.add_argument('--has_norm', default='True', help='Normalize.')
    parser.add_argument('--has_entropy_loss', default='False', help='Has Cross Entropy loss.')
    parser.add_argument('--has_weight_loss', default='False', help='Has Weight Loss.')
    parser.add_argument('--has_v', default='True', help='Has Visual Features.')
    parser.add_argument('--has_a', default='True', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='True', help='Has Textual Features.')

    parser.add_argument('--is_pruning', default='False', help='Pruning Mode')
    parser.add_argument('--weight_mode', default='confid', help='Weight mode')
    parser.add_argument('--fusion_mode', default='concat', help='Fusion mode')
    parser.add_argument('--reattn', default='False', help='reattention')
    parser.add_argument('--central', type=str, default='central_item', help='reattention')
    parser.add_argument('--NA', action='store_true', default=False, help='node alignment.')
    parser.add_argument('--LGN', action='store_true', default=False, help='lightGCN.')
    args = parser.parse_args()
    
    seed = args.seed
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ##########################################################################################################################################
    data_path = args.data_path
    save_file = args.save_file

    learning_rate = args.l_r
    weight_decay = args.weight_decay
    alpha_threshold = args.alpha_threshold
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    num_routing = args.num_routing
    topK = args.topK
    prefix = args.prefix
    aggr_mode = args.aggr_mode
    dropout = args.dropout
    weight_mode = args.weight_mode
    fusion_mode = args.fusion_mode
    has_act = True if args.has_act == 'True' else False
    pruning = True if args.is_pruning == 'True' else False
    has_norm = True if args.has_norm == 'True' else False
    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False
    has_entropy_loss = True if args.has_entropy_loss == 'True' else False
    has_weight_loss = True if args.has_weight_loss == 'True' else False
    dim_E = args.dim_E
    dim_C = None if args.dim_C == 0 else args.dim_C
    is_word = True if data_path == 'Tiktok' else False
    writer = None#SummaryWriter()
    reattn = True if args.reattn == 'True' else False
    central = args.central
    # with open(data_path+'/result/result{0}_{1}.txt'.format(l_r, weight_decay), 'w') as save_file:
    #     save_file.write('---------------------------------lr: {0} \t Weight_decay:{1} ---------------------------------\r\n'.format(l_r, weight_decay))
    ##########################################################################################################################################
    TIMESTAMP = "{0:%m-%d/T%H-%M-%S}".format(datetime.now())
    log_name = './LOG/' + TIMESTAMP + '.log'
    logger = Logger(log_name)
    logger.write(args.__repr__())
    s_time = datetime.now()
    
    print('Data loading ...')

    num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat = data_load(data_path)


    train_dataset = TrainingDataset(num_user, num_item, user_item_dict, train_edge)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)#, num_workers=num_workers)

    val_data = np.load('./Data/'+data_path+'/val_full.npy', allow_pickle=True)
    test_data = np.load('./Data/'+data_path+'/test_full.npy', allow_pickle=True)
    print('Data has been loaded.')
    if args.NA:
        print('computing NA edge')
        mm_edge = edgeNA(num_user, num_item, train_edge)
        print('NA edge is obtained')
    else:
        mm_edge = None
    ##########################################################################################################################################
    if args.NA:
        model = NA_NET(num_user, num_item, train_edge, user_item_dict, weight_decay, 
                        v_feat, a_feat, t_feat, 
                        aggr_mode, weight_mode, fusion_mode,
                        num_routing, dropout, 
                        has_act, has_norm, has_entropy_loss, has_weight_loss,
                        is_word, alpha_threshold, reattn, central,
                        dim_E, dim_C,
                        pruning, mm_edge).cuda()
    else:
        model = Net(num_user, num_item, train_edge, user_item_dict, weight_decay, 
                        v_feat, a_feat, t_feat, 
                        aggr_mode, weight_mode, fusion_mode,
                        num_routing, dropout, 
                        has_act, has_norm, has_entropy_loss, has_weight_loss,
                        is_word, alpha_threshold, reattn, central,
                        dim_E, dim_C,
                        pruning).cuda()
    ##########################################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])
    ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    num_decreases = 0 
    for epoch in range(num_epoch):
        loss = train(epoch, len(train_dataset), train_dataloader, model, optimizer, batch_size, logger)
        if torch.isnan(loss):
            with open('./Data/'+data_path+'/result_{0}.txt'.format(save_file), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} is Nan'.format(learning_rate, weight_decay))
            logger.write('lr: {0} \t Weight_decay:{1} is Nan'.format(learning_rate, weight_decay))
            break
        torch.cuda.empty_cache()

        val_precision, val_recall, val_ndcg = full_t(epoch, model, 'Train', logger)
        val_precision, val_recall, val_ndcg = full_vt(epoch, model, val_data, 'Val', logger)
        test_precision, test_recall, test_ndcg = full_vt(epoch, model, test_data, 'Test', logger)

        if test_recall > max_recall:
            max_precision = test_precision
            max_recall = test_recall
            max_NDCG = test_ndcg
            num_decreases = 0
        else:
            if num_decreases > 20:
                #with open('./Data/'+data_path+'/GRCN_result_{0}.txt'.format(save_file), 'a') as save_file:
                #    save_file.write('dropout: {0} \t lr: {1} \t Weight_decay:{2} =====> Precision:{3} \t Recall:{4} \t NDCG:{5}\r\n'.
                #                    format(dropout, learning_rate, weight_decay, max_precision, max_recall, max_NDCG))
                logger.write('BEST RESULT: lr: {0} \t alpha_threshold:{1}=====> Precision:{2} \t Recall:{3} \t NDCG:{4}\r\n'.
                                    format(learning_rate, alpha_threshold, max_precision, max_recall, max_NDCG))
                #break
            else:
                num_decreases += 1
                
        if epoch != 0 and epoch%50 == 0:
            learning_rate = learning_rate * 0.8
            optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])
            print('=====================decreasing lr to:', learning_rate)

