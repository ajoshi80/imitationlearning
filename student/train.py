import torch
import torch.optim as optim
import torchvision.models as models
import argparse
from pong_dataset import PongDataset
from net import DQN
import torch.nn as nn
import gym
from replay_buffer import ReplayBuffer
from wrappers import PreproWrapper, MaxAndSkipEnv
from preprocess import greyscale
import numpy as np

def soft_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def evaluate_in_environment(net):
    env = gym.make("Pong-v0")
    env = MaxAndSkipEnv(env, skip=4)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=True)
    evaluate(net, env=env)


def evaluate(net, env=None, num_episodes=50):
    """
    Evaluation with same procedure as the training
    """
    print("Evaluating...")
    # arguments defaults

    # replay memory to play
    replay_buffer = ReplayBuffer(1000000, 4)
    rewards = []

    for i in range(num_episodes):
        total_reward = 0
        state = env.reset()
        while True:

            # store last state in buffer
            idx     = replay_buffer.store_frame(state)
            q_input = replay_buffer.encode_recent_observation()

            action = net.get_action(q_input)

            # perform action in env
            new_state, reward, done, info = env.step(action)

            # store in replay memory
            replay_buffer.store_effect(idx, action, reward, done)
            state = new_state

            # count reward
            total_reward += reward
            if done:
                break

        # updates to perform at the end of an episode
        rewards.append(total_reward)     

    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

    if num_episodes > 1:
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    print(msg)
    return avg_reward





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
            outputs = net(inputs)
            loss = soft_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        torch.save(net, "student_epoch_"+str(epoch))

        evaluate_in_environment(net)


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