from mindspore.nn.layer.normalization import _BatchNorm


def disable_running_stats(model):
    for _, cell in model.cells_and_names():
        if isinstance(cell, _BatchNorm):
            cell.backup_momentum = cell.momentum
            cell.momentum = 0

def enable_running_stats(model):
    for _, cell in model.cells_and_names():
        if isinstance(cell, _BatchNorm) and hasattr(cell, "backup_momentum"):
            cell.momentum = cell.backup_momentum