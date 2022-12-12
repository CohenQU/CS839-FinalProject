import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import sys
import argparse
from otnae1 import *

def create_eval_dataset(dataset, num_of_samples):
    size = dataset.shape[0]
    if num_of_samples != -1:
        index = np.random.choice(range(size), num_of_samples, replace=False)
    else:
        index = range(size)
    eval_dataset = dataset[index, :]
    return eval_dataset

def plot_2Ddist(dataset, title, filepath, save, index_X, index_Y):
    plt.clf()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.scatter(dataset[:, index_X], dataset[:, index_Y], alpha=0.2, c='#66c2a5', s=3)
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])
    plt.xlable('dim-{}'.format(index_X))
    plt.ylable('dim-{}'.format(index_Y))
    plt.title(title)
    if save:
        plt.savefig(filepath, dpi=400)
    else:
        plt.show()

colors = [
    "#8dd3c7", "#ffffb3", "#bebada",
    "#fb8072", "#80b1d3", "#fdb462",
    "#b3de69", "#fccde5", "#d9d9d9"
]
num_of_colors = len(colors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--native_dim", help="dimensionality of native space", type=int, required=True)
    parser.add_argument("--encoder_model", type=str, default="VanillaAE")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--hidden_layer", type=int, help="number of nodes in the hidden layer", default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", help="the path of data generated in native space", type=str, required=True)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--num_of_samples",type=int,default=1000)

    args = parser.parse_args()
    print(args)
    folder_name = "models"

    env_id = args.env_id
    native_dim = args.native_dim
    encoder_model = args.encoder_model
    model_path = args.model_path
    hidden_layer = args.hidden_layer
    batch_size = args.batch_size
    data_path = args.data_path
    save_path = args.save_path
    num_of_samples = args.num_of_samples
    
    dataset = np.load(data_path)
    dataset = dataset.reshape((-1, native_dim))
    eval_dataset = create_eval_dataset(dataset, num_of_samples)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    os.makedirs(save_path, exist_ok=True)

    model = OTNAE(native_dim, hidden_layer)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    original_stdout = sys.stdout
    fig = plt.figure()
    for index_X in range(native_dim-1):
        for index_Y in range(index_X+1, native_dim):
            parent_dir = os.path.join(save_path, "dim{}-vs-dim{}".format(index_X, index_Y))
            try:
                os.makedirs(parent_dir, exist_ok=True)
                # print("Directory '%s' created successfully" %path)
            except OSError as error:
                # print("Directory '%s' can not be created")
                pass

            plot_2Ddist(eval_dataset, "{} Original Action Distribution ({} vs {})".format(env_id, index_X, index_Y), os.path.join(parent_dir, "original.png"), save=True, index_X=index_X, index_Y=index_Y)

            recon_data = [None for i in range(native_dim)]
            output_data = [None for i in range(native_dim)]
            loss = [0 for i in range(native_dim)]
            for _, input_data in enumerate(eval_dataloader, 0):
                encoded = model.encoder(input_data)
                num_of_inputs = input_data.shape[0]
                outputs = torch.zeros(native_dim, num_of_inputs, native_dim)
                index = 0
                for decoder in model.decoders:
                    for i in range(index, native_dim):
                        outputs[i, :, :] += decoder(encoded[:, index].reshape(-1, 1))
                    index += 1
                for i in range(native_dim):
                    output_data[i] = outputs[i]
                    if recon_data[i] == None:
                        recon_data[i] = output_data[i].detach().clone()
                    else:
                        recon_data[i] = torch.cat((recon_data[i], output_data[i].detach().clone()), 0)
                    loss[i] += (torch.nn.MSELoss()(output_data[i], input_data)).item() * len(input_data)

            with open(os.path.join(parent_dir, "loss.txt"), 'w') as f:
                sys.stdout = f
                for i in range(native_dim):
                    print("loss of using {}D encoded data: {}".format(i+1, loss[i]/len(eval_dataset)))
                sys.stdout = original_stdout

            for i in range(native_dim-2):
                plt.clf()
                fig.patch.set_facecolor('white')
                for j in range(i+2, i-1, -1):
                    plt_data = recon_data[j].numpy()
                    color = colors[j%num_of_colors]
                    label = "{}D recon".format(j+1)
                    plt.scatter(plt_data[:, index_X], plt_data[:, index_Y], alpha=0.4, c=color, s=3, label=label)
                plt.xlim([-1.0, 1.0])
                plt.ylim([-1.0, 1.0])
                plt.xlabel('dim-{}'.format(index_X))
                plt.ylabel('dim-{}'.format(index_Y))
                plt.title("{} Reconstruction Space ({} vs {})".format(env_id, index_X, index_Y))
                legend = plt.legend(loc="upper right", markerscale=3)
                filepath = os.path.join(parent_dir, "{}dims-vs-{}dims-vs-{}dims.png".format(i+1, i+2, i+3))
                plt.savefig(filepath, dpi=400)
            
            for i in range(native_dim-1):
                plt.clf()
                fig.patch.set_facecolor('white')
                plt.scatter(eval_dataset[:, index_X], eval_dataset[:, index_Y], alpha=0.4, c=colors[native_dim%num_of_colors], s=3, label="original")
                for j in range(i+1, i-1, -1):
                    plt_data = recon_data[j].numpy()
                    color = colors[j%num_of_colors]
                    label = "{}D recon".format(j+1)
                    plt.scatter(plt_data[:, index_X], plt_data[:, index_Y], alpha=0.4, c=color, s=3, label=label)
                
                plt.xlim([-1.0, 1.0])
                plt.ylim([-1.0, 1.0])
                plt.xlable('dim-{}'.format(index_X))
                plt.ylable('dim-{}'.format(index_Y))
                plt.title("{} Reconstruction Space ({} vs {})".format(env_id, index_X, index_Y))
                legend = plt.legend(loc="upper right", markerscale=3)
                filepath = os.path.join(parent_dir, "{}dims-vs-{}dims.png".format(i+1, i+2))
                plt.savefig(filepath, dpi=400)