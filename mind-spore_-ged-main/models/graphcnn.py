import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import sys
import numpy as np
from mindspore.common.initializer import initializer, XavierUniform
sys.path.append("models/")
from mlp import MLP
import msadapter.pytorch as torch

ms.set_context(device_target='GPU')

class GraphCNN(nn.Cell):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim,
                 final_dropout, num_nodes, edge_mat, neighbor_pooling_type):
        """
            num_layers: number of layers in GNN (INCLUDING the input layer)
            num_mlp_layers: number of layers in MLP (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units in ALL layers
            output_dim: number of emotion categories
            final_dropout: dropout ratio after the final linear prediction layer
            num_nodes: number of nodes in the emotion-brain bipartite graph
            edge_mat: a torch long tensor containing the edge list, will be used to create a torch sparse tensor
            neighbor_pooling_type: how to aggregate neighboring nodes (sum or average)
            device: which device to use
        """

        super(GraphCNN, self).__init__(auto_prefix=True)

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.final_dropout = final_dropout
        self.num_nodes = num_nodes
        self.edge_mat = edge_mat
        self.neighbor_pooling_type = neighbor_pooling_type

        self.emotion_embeddings = ms.Parameter(initializer(XavierUniform(), [1, output_dim, input_dim]))

        # a list of MLPs
        self.mlps = nn.CellList()

        # a list of batch norms applied to the output of MLP (input of the final linear prediction layer)
        self.batch_norms = nn.CellList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # a linear function that maps the emotion representation at each layer into a prediction score
        self.linears_prediction = nn.CellList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Dense(input_dim, 1))
            else:
                self.linears_prediction.append(nn.Dense(hidden_dim, 1))

    def __preprocess_neighbors_sumavepool(self, batch_size):
        # create an adjacency matrix (a block-diagonal sparse matrix) for concatenated (batch) graph

        edge_mat_list = []
        start_idx = [0]
        for i in range(batch_size):
            start_idx.append(start_idx[i] + self.num_nodes)
            edge_mat_list.append(self.edge_mat + start_idx[i])
        Adj_block_idx = ops.cat(edge_mat_list, 1)
        Adj_block_elem = ops.ones(Adj_block_idx.shape[1])

        # add self-loops in the adjacency matrix, aggregate the central node and neighboring nodes altogether
        batch_nodes = start_idx[-1]
        self_loop_edge = ms.Tensor([range(batch_nodes), range(batch_nodes)],dtype=ms.int64)
        elem = ops.ones(batch_nodes)
        Adj_block_idx = ops.cat((Adj_block_idx, self_loop_edge), 1)
        Adj_block_elem = ops.cat((Adj_block_elem, elem), 0)
        # 下面的代码存在问题，在pytorch源码中这步的操作应该是构造稀疏矩阵，但是ms里没有相应的操作，期待解决
        Adj_block = ops.dense_to_sparse_coo(ms.Tensor([Adj_block_idx, Adj_block_elem]), shape=(start_idx[-1], start_idx[-1]))
        return Adj_block

    def next_layer(self, h, layer, Adj_block=None): 
        # get the representation for the next layer

        # aggregate the central node and neighboring nodes altogether

        pooled = Adj_block@h
        if self.neighbor_pooling_type == "average":
            # average pooling
            degree = Adj_block@ops.ones((Adj_block.shape[0], 1))
            pooled = pooled / degree

        # representation after aggregation (and MLP mapping)
        pooled_rep = self.mlps[layer](pooled)

        # batch normalization
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = ops.ReLU(h)

        return h

    def construct(self, batch_feature):
        batch_size = batch_feature.shape[0]
        batch_feature[:, 0:self.output_dim, :] = ops.tile(self.emotion_embeddings,(batch_size,1,1))
        X_concat = batch_feature.reshape(-1, batch_feature.shape[2])


        Adj_block = self.__preprocess_neighbors_sumavepool(batch_size)

        # a list of hidden representations at different layers (including the input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers-1):
            h = self.next_layer(h, layer, Adj_block=Adj_block)
            hidden_rep.append(h)

        score_over_layer = 0

        # extract emotion representations and predict emotion scores
        for layer, h in enumerate(hidden_rep):
            h_emotion = ops.cat([h[i*self.num_nodes:i*self.num_nodes+self.output_dim] for i in range(batch_size)], 0)
            score_over_layer += ops.dropout(self.linears_prediction[layer](h_emotion), self.final_dropout, training=self.training)
        score_over_layer = ops.sigmoid(score_over_layer)

        return score_over_layer.reshape(batch_size, self.output_dim)
