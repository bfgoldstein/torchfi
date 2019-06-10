def countZeroWeights(model):
    zeros = 0
    nnz = 0
    for param in model.parameters():
        if param is not None:
            zeros += param.numel() - param.nonzero().size(0)
            nnz += param.nonzero().size(0)
    sparsity = zeros / float(zeros + nnz)
    return zeros, nnz, sparsity