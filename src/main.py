import argparse
import time
import math

import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
# import seaborn as sns
import data
import model
import os
import os.path as osp

parser = argparse.ArgumentParser(description='PyTorch Language Model')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
# you can increase the seqence length to see how well the model works when capturing long-term dependencies
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')
parser.add_argument('--data_dir', type=str, default='../data/wikitext2', help='dataset path')
parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='model save path')

# feel free to add some other arguments
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}

# anylearn
# use_gpu = True
# data_dir = args.data_dir + "/wikitext2/"  ## You need to specify the data_dir first
# data_loader = data.Corpus(data_dir, batch_size, args.max_sql)

# local
use_gpu = torch.cuda.is_available()
data_dir = args.data_dir
data_loader = data.Corpus("../data/wikitext2", batch_size, args.max_sql)

lr = 5.0

if use_gpu:
    # torch.cuda.set_device(args.gpu_id)
    # device = torch.device(args.gpu_id)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# WRITE CODE HERE within two '#' bar                                                           #
# Build model, optimizer and so on                                                             #
################################################################################################

# model = model.RNN(len(data_loader.vocabulary), ninput=1000, nhid=256, nlayers=1,rnnType="LSTM",device=device)
model = model.RNN(len(data_loader.vocabulary), ninput=1000, nhid=256, nlayers=6, rnnType="Transformer", device=device,
                  args=args)


# model = torch.load("../checkpoint/best_model.pt", map_location=torch.device('cpu'))

# model.half()

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)


################################################################################################
# def grad_clipping(rnn_net, theta):
#     if isinstance(rnn_net, nn.Module):
#         params = [p for p in rnn_net.parameters() if p.requires_grad]
#     else:
#         params = rnn_net.params
#     norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
#     if norm > theta:
#         for param in params:
#             param.grad[:] *= theta / norm

# WRITE CODE HERE within two '#' bar                                                           #
# Evaluation Function                                                                          #
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       #
################################################################################################
def evaluate(epoch, device, dataloader, model, criterion):
    dataloader.set_valid()
    model.eval()
    end_flag = False
    l_sum, n, start = 0.0, 0, time.time()
    total_correct = 0.0
    index = 0
    while not end_flag:
        x, Y, end_flag = dataloader.get_batch()
        x = x.to(device)
        Y = Y.to(device)
        output, state = model(x)
        output = output.view(-1, output.shape[-1])

        _, predictions = torch.max(output, 1)
        total_correct += torch.sum(predictions == Y)
        l = criterion(output, Y.long()).mean()
        l_sum += l.item() * Y.shape[0]
        n += Y.shape[0]

        # 绘制attention热力图
        # x_ = model.drop(model.embed(x))
        # attention_maps = model.rnn.transformer_encoder.get_attention_maps(x_)
        # attention = attention_maps[0][0]
        # word_index = x[:, 0]
        # word_list = []
        # for i in word_index:
        #     word_list.append(dataloader.vocabulary[i])
        #
        # attention = attention.detach().numpy()
        # attention = pd.DataFrame(attention, index=word_list, columns=word_list)
        # sns.heatmap(attention)
        # # plt.show()
        # plt.savefig(f"../result/heatmap_{index}.png")
        # plt.close()
        # index += 1


    try:
        perplexity = math.exp(l_sum / n)
    except OverflowError:
        perplexity = float('inf')

    epoch_acc = total_correct.float() / n

    print('valid epoch %d, perplexity %f, accuracy %f, time %.2f sec' % (
        epoch, perplexity, epoch_acc, time.time() - start))

    return perplexity, epoch_acc.item()


################################################################################################


# WRITE CODE HERE within two '#' bar                                                           #
# Training Function                                                                            #     
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       # 
################################################################################################
def train(epoch, device, dataloader, model, criterion, optimizer):
    dataloader.set_train()
    model.train()
    end_flag = False
    l_sum, n, start = 0.0, 0, time.time()
    total_correct = 0.0
    while not end_flag:
        x, Y, end_flag = dataloader.get_batch()
        x = x.to(device)
        Y = Y.to(device)

        output, state = model(x)

        output = output.view(-1, output.shape[-1])

        _, predictions = torch.max(output, 1)
        total_correct += torch.sum(predictions == Y)

        l = criterion(output, Y.long()).mean()
        optimizer.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # grad_clipping(model, 0.1)

        optimizer.step()
        l_sum += l * Y.numel()
        n += Y.shape[0]




    try:
        perplexity = math.exp(l_sum / n)
    except OverflowError:
        perplexity = float('inf')

    epoch_acc = total_correct.float() / n

    print('train epoch %d, perplexity %f, accuracy %f, time %.2f sec' % (
        epoch, perplexity, epoch_acc, time.time() - start))
    print("第%d轮的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

    return perplexity, epoch_acc.item()


################################################################################################


# WRITE CODE HERE within two '#' bar                                                           #
# Loop over epochs                                                                             #
################################################################################################
# 绘图所需
best_acc = 0.0
best_model = None
epoch_train_perplexity = []
epoch_train_accuracy = []
epoch_valid_perplexity = []
epoch_valid_accuracy = []
num_epochs_array = [i + 1 for i in range(args.epochs)]

for epoch in range(1, args.epochs + 1):
    train_perplexity, train_acc = train(epoch=epoch, device=device, dataloader=data_loader, model=model,
                                        criterion=criterion, optimizer=optimizer)
    valid_perplexity, valid_acc = evaluate(epoch=epoch, device=device, dataloader=data_loader, model=model,
                                           criterion=criterion)
    scheduler.step()
    epoch_train_perplexity.append(train_perplexity)
    epoch_train_accuracy.append(train_acc)
    epoch_valid_perplexity.append(valid_perplexity)
    epoch_valid_accuracy.append(valid_acc)

    if valid_acc > best_acc:
        best_acc = valid_acc
        best_model = model
        torch.save(best_model, os.path.join(args.save_dir, 'best_model.pt'))

# 绘图
# 绘制训练曲线图
plt.figure()
# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(121)
plt.xlabel('epochs')  # x轴标签
plt.ylabel('perplexity')  # y轴标签
# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(num_epochs_array, epoch_train_perplexity, linewidth=1, linestyle="solid", label="train perplexity")
plt.plot(num_epochs_array, epoch_valid_perplexity, linewidth=1, linestyle="solid", label="valid perplexity",
         color='black')
plt.legend()
plt.title('Perplexity curve')

plt.subplot(122)
plt.xlabel('epochs')  # x轴标签
plt.ylabel('accuracy')  # y轴标签

# 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 增加参数color='red',这是红色。
plt.plot(num_epochs_array, epoch_train_accuracy, color='red', linewidth=1, linestyle="solid", label="train acc")
plt.plot(num_epochs_array, epoch_valid_accuracy, color='orange', linewidth=1, linestyle="solid", label="valid acc")
plt.legend()
plt.title('Accuracy curve')

plt.savefig("./checkpoints/result.png")

print("最高验证集精确度:", best_acc)

################################################################################################
