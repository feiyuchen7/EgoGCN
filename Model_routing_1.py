import math
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch_scatter
from SAGEConv import SAGEConv
from GATConv import GATConv
from GATConv_ori import GATConv_ori
from torch_geometric.utils import add_self_loops, dropout_adj#, scatter_
from GCNConv import GCNConv
# from torch.utils.checkpoint import checkpoint
##########################################################################

def scatter_(name, src, index, dim_size=None):
     assert name in ['add', 'mean', 'max']
     op = getattr(torch_scatter, 'scatter_{}'.format(name))
     fill_value = -1e9 if name == 'max' else 0
     out = op(src, index, 0, None, dim_size)
     if isinstance(out, tuple):
         out = out[0]
     if name == 'max':
        out[out == fill_value] = 0
     return out

class EGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm):
        super(EGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.aggr_mode = aggr_mode
        self.has_act = has_act
        self.has_norm = has_norm
        self.id_embedding = nn.Parameter( nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))))
        self.conv_embed_1 = GCNConv(dim_E, dim_E, aggr=aggr_mode)         
        self.conv_embed_2 = GCNConv(dim_E, dim_E, aggr=aggr_mode)
        #self.conv_embed_3 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding   
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)  
        if self.has_norm:
            x = F.normalize(x) 
        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector)      

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector) 
        if self.has_act:
            x_hat_2 = F.leaky_relu_(x_hat_2)
        #x_hat_3 = self.conv_embed_3(x_hat_2, edge_index, weight_vector)
        return x + x_hat_1 + x_hat_2 #+1/4*x_hat_3  

class RealGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm):
        super(RealGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.aggr_mode = aggr_mode
        self.has_act = has_act
        self.has_norm = has_norm
        self.id_embedding = nn.Parameter( nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))))
        self.conv_embed_1 = GCNConv(dim_E, dim_E, aggr=aggr_mode)         
        self.conv_embed_2 = GCNConv(dim_E, dim_E, aggr=aggr_mode)
        self.conv_embed_3 = GCNConv(dim_E, dim_E, aggr=aggr_mode)

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding   
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)  
        if self.has_norm:
            x = F.normalize(x) 

        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector)       

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector) 
        if self.has_act:
            x_hat_2 = F.leaky_relu_(x_hat_2)
        x_hat_3 = self.conv_embed_3(x_hat_1, edge_index, weight_vector) 
        return x + x_hat_1 + x_hat_2 + x_hat_3  


'''class CGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm, alpha_threshold, reattn, central, is_word=False):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing
        self.has_act = has_act
        self.has_norm = has_norm
        self.dim_C = dim_C
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_C))))
        self.conv_embed_1 = GATConv(self.dim_C, self.dim_C, alpha_threshold, reattn, central)
        

    def forward(self, features, features2, features3, edge_index):

        if self.has_norm:
            preference = F.normalize(self.preference)     #normalize preference   (36656,64)
            features = F.normalize(features)              #normalize item features(76085,64)
            features2 = F.normalize(features2)
            features3 = F.normalize(features3)

        for i in range(self.num_routing):                 #equation 4
            x =  torch.cat((preference, features), dim=0)
            y =  torch.cat((preference, features2), dim=0)
            z =  torch.cat((preference, features3), dim=0)
            x_hat_1 = self.conv_embed_1(x, y, z, edge_index)   #(112741,64)
            preference = preference + x_hat_1[:self.num_user]   

            if self.has_norm:
                preference = F.normalize(preference)

        x =  torch.cat((preference, features), dim=0)      #input both user embedding (preference) and item embedding (features)
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)
        x_hat_1 = self.conv_embed_1(x, y, z, edge_index)         

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)   #[112741, 64]
        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1)  '''


class CGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm, alpha_threshold, reattn, central, is_word=False):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing
        self.has_act = has_act
        self.has_norm = has_norm
        self.dim_C = dim_C
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_C))))
        self.conv_embed_1 = GATConv(self.dim_C, self.dim_C, alpha_threshold, reattn, central)
        self.conv_embed_2 = GATConv_ori(self.dim_C, self.dim_C)
        self.convGCN_2 = GCNConv(self.dim_C, self.dim_C, aggr='add')  

    def forward(self, features, features2, features3, edge_index):

        if self.has_norm:
            preference = F.normalize(self.preference)     #normalize preference   (36656,64)
            features = F.normalize(features)              #normalize item features(76085,64)
            features2 = F.normalize(features2)
            features3 = F.normalize(features3)

        #for i in range(self.num_routing):                 
        x =  torch.cat((preference, features), dim=0)
        y =  torch.cat((preference, features2), dim=0)
        z =  torch.cat((preference, features3), dim=0)
        '''x_hat_1 = self.conv_embed_1(x, y, z, edge_index)   #(112741,64)
        preference = preference + x_hat_1[:self.num_user]   

        if self.has_norm:
            preference = F.normalize(preference)

        x =  torch.cat((preference, features), dim=0)'''      #input both user embedding (preference) and item embedding (features)
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)
        
        for i in range(1):   #fusion
            x_hat_1 = self.conv_embed_1(x, y, z, edge_index)         
            #x = x_hat_1       
            x = x + x_hat_1    
            if self.has_norm:
                x = F.normalize(x)
        for i in range(self.num_routing):       #routing
            x_hat_1 = self.conv_embed_2(x, edge_index)
            x = x + x_hat_1
            if self.has_norm:
                x = F.normalize(x)

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)   #[112741, 64]
        
        return x, self.conv_embed_1.alpha.view(-1, 1)  


class Net(torch.nn.Module):
    def __init__(self, num_user, num_item, edge_index, user_item_dict, reg_weight, 
                        v_feat, a_feat, t_feat, 
                        aggr_mode, weight_mode, fusion_mode,
                        num_routing, dropout, 
                        has_act, has_norm, has_entropy_loss, has_weight_loss,
                        is_word, alpha_threshold, reattn, central,
                        dim_E, dim_C,
                        pruning):
        super(Net, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.weight_mode = weight_mode
        self.fusion_mode = fusion_mode
        self.weight = torch.tensor([[1.0],[-1.0]]).cuda()
        self.reg_weight = reg_weight
        self.dropout = dropout

        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.id_gcn = EGCN(num_user, num_item, dim_E, aggr_mode, has_act, has_norm)
        self.id_gcn_real = RealGCN(num_user, num_item, dim_E, aggr_mode, has_act, has_norm)
        self.v_feat = v_feat
        self.a_feat = a_feat
        self.t_feat = t_feat
        self.has_entropy_loss = has_entropy_loss
        self.has_weight_loss = has_weight_loss
        self.alpha_threshold = alpha_threshold
        self.reattn = reattn
        self.central = central

        self.pruning = pruning
        num_model = 0
        if v_feat is not None:
            self.v_gcn = CGCN(num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm, self.alpha_threshold, self.reattn, self.central)
            num_model += 1

        if a_feat is not None:
            self.a_gcn = CGCN(num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm, self.alpha_threshold, self.reattn, self.central)
            num_model += 1
        
        if t_feat is not None:
            self.t_gcn = CGCN(num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm, self.alpha_threshold, self.reattn, self.central, is_word)
            num_model += 1

        self.model_specific_conf = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user+num_item, num_model)))) 

        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))).cuda()
        self.fc_weight = nn.Linear(3,1)
        self.MLPv = nn.Linear(self.v_feat.size(1), dim_C)
        self.MLPa = nn.Linear(self.a_feat.size(1), dim_C)

        if self.t_feat.size()[0] == 2:
            self.special_t = True
            self.word_tensor = torch.LongTensor(self.t_feat).cuda()
            self.t_feat = nn.Embedding(torch.max(self.t_feat[1])+1, dim_C)
            nn.init.xavier_normal_(self.t_feat.weight)
        else:
            self.special_t = False
            self.MLPt = nn.Linear(self.t_feat.size(1), dim_C)
        

    def forward(self):
        weight = None
        content_rep = None
        num_modal = 0
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)  #[2, 564365]
        
        if self.central == 'central_user':
            edge_index = edge_index[[1,0]]
        features_v = F.leaky_relu(self.MLPv(self.v_feat))   #equation 1
        features_a = F.leaky_relu(self.MLPa(self.a_feat))
        if self.special_t:
            features_t = scatter_('mean', self.t_feat(self.word_tensor[1]), self.word_tensor[0]).cuda()
        else:
            features_t = F.leaky_relu(self.MLPt(self.t_feat))

        if self.v_feat is not None:
            num_modal += 1
            v_rep, weight_v = self.v_gcn(features_v, features_a, features_t, edge_index)
            weight = weight_v   #[1128730, 1]
            content_rep = v_rep #[112741, 64]

        if self.a_feat is not None:
            num_modal += 1
            a_rep, weight_a = self.a_gcn(features_a, features_v, features_t, edge_index)
            if weight is  None:
                weight = weight_a  
                content_rep = a_rep
            else:
                content_rep = torch.cat((content_rep,a_rep),dim=1) #[112741, 128]
                if self.weight_mode == 'mean':
                    weight = weight+ weight_a
                else:
                    weight = torch.cat((weight, weight_a), dim=1)  #[1128730, 2]

        if self.t_feat is not None:
            num_modal += 1
            t_rep, weight_t = self.t_gcn(features_t, features_a, features_v, edge_index)
            if weight is None:
                weight = weight_t   
                conetent_rep = t_rep
            else:
                content_rep = torch.cat((content_rep,t_rep),dim=1)  #[112741, 192]
                if self.weight_mode == 'mean':  
                    weight  = weight+  weight_t
                else:
                    weight = torch.cat((weight, weight_t), dim=1)   #[1128730, 3]

        if self.weight_mode == 'mean':
        	weight = torch.mean(weight, 1, True)

        elif self.weight_mode == 'max':
        	weight, _ = torch.max(weight, dim=1)
        	weight = weight.view(-1, 1)
            
        elif self.weight_mode == 'confid':
            confidence = torch.cat((self.model_specific_conf[edge_index[0]], self.model_specific_conf[edge_index[1]]), dim=0) 
            weight = weight * confidence   #[1128730,3] * [1128730,3] -> [1128730,3]
            weight, _ = torch.max(weight, dim=1)  
            weight = weight.view(-1, 1)    #[1128730, 1]
        

        if self.pruning:
            weight = torch.relu(weight)   

        if self.weight_mode == 'GCN':
            id_rep = self.id_gcn_real(edge_index, weight)   #[112741, 64]
        else:
            id_rep = self.id_gcn(edge_index, weight)   #[112741, 64]

        if self.fusion_mode == 'concat':
            representation = torch.cat((id_rep, content_rep), dim=1)
        elif self.fusion_mode  == 'id':
            representation = id_rep
        elif self.fusion_mode == 'mean':
            representation = (id_rep+v_rep+a_rep+t_rep)/4
        elif self.fusion_mode == 'multimodal':
            representation = content_rep

        self.result = representation
        return representation

    def loss(self, user_tensor, item_tensor):
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)
        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score*item_score, dim=1).view(-1, 2)   
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))

        if self.weight_mode == 'GCN':
            reg_embedding_loss = (self.id_gcn_real.id_embedding[user_tensor]**2 + self.id_gcn_real.id_embedding[item_tensor]**2).mean() 
        else:
            reg_embedding_loss = (self.id_gcn.id_embedding[user_tensor]**2 + self.id_gcn.id_embedding[item_tensor]**2).mean() 
        reg_content_loss = torch.zeros(1).cuda() 
        if self.v_feat is not None:
            reg_content_loss = reg_content_loss + (self.v_gcn.preference[user_tensor]**2).mean()
        if self.a_feat is not None:
            reg_content_loss = reg_content_loss + (self.a_gcn.preference[user_tensor]**2).mean()
        if self.t_feat is not None:            
            reg_content_loss = reg_content_loss + (self.t_gcn.preference[user_tensor]**2).mean()

        #reg_confid_loss = (self.model_specific_conf**2).mean()
        
        reg_loss = reg_embedding_loss + reg_content_loss


        reg_loss = self.reg_weight * reg_loss
        return loss+reg_loss, loss, reg_embedding_loss+reg_content_loss, reg_embedding_loss, reg_content_loss


    def accuracy(self, step=2000, topk=10): 
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        start_index = 0
        end_index = self.num_user if step==None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.num_user), dim=0)
            start_index = end_index
            
            if end_index+step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = self.num_user      
        precision = recall = ndcg = 0.0

        for row, col in self.user_item_dict.items():
            user = row
            pos_items = set(col)
            num_pos = len(pos_items)
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)

            num_hit = len(pos_items.intersection(items))
            
            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            if max_ndcg_score == 0:
                continue
                
            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)

            ndcg += ndcg_score/max_ndcg_score

        return precision/length, recall/length, ndcg/length



    def full_accuracy(self, val_data, step=100, topk=10):
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        start_index = 0
        end_index = self.num_user if step==None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))-self.num_user
                    score_matrix[row][col] = 1e-5

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.num_user), dim=0)
            start_index = end_index
            
            if end_index+step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = 0#len(val_data)        
        precision = recall = ndcg = 0.0

        for data in val_data:
            user = data[0]
            pos_items = set(data[1:])
            num_pos = len(pos_items)
            if num_pos == 0:
                continue
            length += 1
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)

            num_hit = len(pos_items.intersection(items))
            
            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            if max_ndcg_score == 0:
                continue
                
            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)

            ndcg += ndcg_score/max_ndcg_score

        return precision/length, recall/length, ndcg/length
