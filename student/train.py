import torch
import torch.optim as optim
import torchvision.models as models
import argparse
from pong_dataset import PongDataset
from net import DQN
import torch.nn as nn

def evaluate():
    running_loss = 0.0
    for i,batch in enumerate(test_loader):
        inputs, labels = batch
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%5d] validation loss: %.3f' %
                ( i + 1, running_loss / 2000))
            running_loss = 0.0
        


def train(args):
    pong_dataset = PongDataset(args.npz_file, ".")
    trainloader = torch.utils.data.DataLoader(pong_dataset, batch_size=args.batch_size,
                                          shuffle=True)

    criterion = nn.CrossEntropyLoss()
    if args.use_resnet:
        net = models.resnet18(pretrained = True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, args.num_actions)
    else:
        net=DQN(n_actions=args.num_actions)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("Epoch Starting")
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["image"], data["label"]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            print("BEFORE NET CALL")
            outputs = net(inputs)
            print("AFTER NET CALL")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        torch.save(net, "student_epoch_"+str(epoch))

    print('Finished Training')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--npz_file', default="res_demos.npz")
    parser.add_argument("--epochs", default=2)
    parser.add_argument("--num_actions", default=6)
    parser.add_argument("--use_resnet", default=False)
    args = parser.parse_args()
    train(args)