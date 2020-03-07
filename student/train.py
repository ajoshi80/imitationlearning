import torch
import torch.optim as optim
import torchvision.models as models

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
    pong_dataset = PongDataset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net = models.resnet18(pretrained = True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, args.num_actions)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("Epoch Starting")
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
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