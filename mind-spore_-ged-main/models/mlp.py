import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

ms.set_context(device_target='GPU')

# MLP with linear output
class MLP(nn.Cell):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            num_layers: number of layers in MLP (EXCLUDING the input layer). If num_layers=1, it degenerates into a linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units in ALL layers
            output_dim: number of classes for prediction
        """

        super(MLP, self).__init__(auto_prefix=True)

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # linear model
            self.linear = nn.Dense(input_dim, output_dim)
        else:
            # multi-layer model
            self.linear_or_not = False
            self.linears = nn.CellList()
            self.batch_norms = nn.CellList()

            self.linears.append(nn.Dense(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Dense(hidden_dim, hidden_dim))
            self.linears.append(nn.Dense(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def construct(self, x):
        if self.linear_or_not:
            # linear model
            return self.linear(x)
        else:
            # MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = ops.ReLU(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
