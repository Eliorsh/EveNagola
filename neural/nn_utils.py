import torch
import numpy as np


def dataset_accuracy(tensor_dataset, targets, net, loss_func, cuda_device='cpu',
                     v=False):
    net.to(cuda_device)
    net.eval()
    output = net(tensor_dataset.cuda())

    # loss = loss_func(output,torch.LongTensor(targets).cuda())
    loss = loss_func(output, to_one_hot(torch.LongTensor(targets).cuda()))

    output = output.cpu().data.numpy()
    prediction = np.argmax(output, axis=1)

    n_correct = len(np.where(targets == prediction)[0])

    n_tp = len(np.where(np.logical_and(targets == 1, prediction == 1))[0])
    n_tn = len(np.where(np.logical_and(targets == 0, prediction == 0))[0])
    n_fp = len(np.where(np.logical_and(targets == 0, prediction == 1))[0])
    n_fn = len(np.where(np.logical_and(targets == 1, prediction == 0))[0])
    recall = 0 if n_tp == 0 else n_tp / (n_tp + n_fn)
    precision = 0 if n_tp == 0 else n_tp / (n_tp + n_fp)
    n_total = len(targets)
    accuracy = n_correct / n_total
    if v:
        print(
            f'accuracy: {accuracy}, recall: {recall}, precision: {precision}, loss: {loss.item()}')
    return accuracy, recall, precision, loss.item()

# use with binary loss functions
def to_one_hot(x, C=2, tensor_class=torch.FloatTensor):
    """ One-hot a batched tensor of shape (B, ...) into (B, C, ...) """
    x_one_hot = tensor_class(x.size(0), C, *x.shape[1:]).zero_()
    if torch.cuda.is_available():
        x_one_hot = x_one_hot.cuda()
    x_one_hot = x_one_hot.scatter_(1, x.unsqueeze(1), 1)
    return x_one_hot