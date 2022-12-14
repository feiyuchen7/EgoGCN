import torch
import torch.nn.functional as F
from torch.nn import Parameter
# from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import numpy


class GATConv_pre(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha_threshold, reattn, central, self_loops=False):
        super(GATConv_pre, self).__init__(aggr='add')#, **kwargs)
        self.self_loops = self_loops
        self.in_channels = in_channels
        self.out_channels = out_channels  #64
        self.alpha_threshold = float(alpha_threshold[1:])
        self.compare = alpha_threshold[0]
        self.reattn = reattn
        self.central = central
        self.fcy = torch.nn.Linear(out_channels, out_channels)
        self.fcz = torch.nn.Linear(out_channels, out_channels)
        #self.param = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.rand((1128730, 3*out_channels))))

    def forward(self, x, y, z, edge_index, size=None):
        #edge_index, _ = remove_self_loops(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=size, x=x, y=y, z=z)


    def message(self, edge_index_i, x_i, x_j, size_i, y_i, y_j, z_i, z_j):  
        #self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
        #self.alpha = softmax(src=self.alpha, index=edge_index_i, num_nodes=size_i)  

        #tmp = self.alpha.unsqueeze(1).expand_as(x_i)
        gating1 = torch.sigmoid(self.fcy(y_i))
        gating2 = torch.sigmoid(self.fcz(z_i))

        if self.central == 'central_item':   
            #-------------------------------sum3-----------------------------------------
            if self.compare == '>':
                x_i = torch.where(tmp>self.alpha_threshold, (x_i+y_i+z_i)/3, x_i)
            elif self.compare == '<':
                x_i = torch.where(tmp<self.alpha_threshold, (x_i+y_i+z_i)/3, x_i)
            elif self.compare == '=':
                x_i = (x_i + gating1*(y_i) + gating2*(z_i))/3

            if self.reattn:
                self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
                self.alpha = softmax(src=self.alpha, index=edge_index_i, num_nodes=size_i)

            return x_j*(self.alpha.view(-1,1))

        elif self.central == 'central_user': 
            #-------------------------------sum3--------------------------------------
            result = torch.where(tmp>self.alpha_threshold, (x_j+y_j+z_j)/3, x_j)
            
            if self.reattn:
                self.alpha = torch.mul(result, x_i).sum(dim=-1)
                self.alpha = softmax(src=self.alpha, index=edge_index_i, num_nodes=size_i)

            return result*self.alpha.view(-1,1)
        # return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        return aggr_out
