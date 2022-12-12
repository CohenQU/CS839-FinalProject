import torch
import numpy as np
from torch.utils.data import DataLoader
from numpy import linalg as LA
import os
import argparse
from otnae3 import *
from ae1 import *
from vae import *
from otnvae import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--native_dim", help="dimensionality of native space", type=int, required=True)
    parser.add_argument("--latent_dim", help="dimensionality of latent space", type=int, required=True)
    parser.add_argument("--encoder_model", type=str, default="VanillaAE")
    parser.add_argument("--hidden_layer", type=int, help="number of nodes in the hidden layer", default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", help="the path of data generated in native space", type=str, required=True)
    parser.add_argument("--save_dir", help="the directory for storing models and results", type=str, required=True, default="./")
    parser.add_argument("--epoch", help="number of training epochs", type=int, default=100)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--num_actions", help="Number of actions to use during training", type=int, default=None)
    parser.add_argument("--num_records", help="Number of records shown on 1 picture", type=int, default=1)

    args = parser.parse_args()
    print(args)
    env_id = args.env_id
    native_dim = args.native_dim
    latent_dim = args.latent_dim
    encoder_model = args.encoder_model
    hidden_layer = args.hidden_layer
    batch_size = args.batch_size
    data_path = args.data_path
    parent_dir = args.save_dir
    epochs = args.epoch
    lr = args.lr
    
    dataset = np.load(data_path).astype(np.float32)
    if len(dataset.shape) > 2:
        dataset = dataset.reshape(-1, native_dim)
    if args.num_actions > 0:
        dataset, _ = torch.utils.data.random_split(dataset, [args.num_actions, len(dataset) - args.num_actions])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    folder_name = "{}_{}".format(env_id,encoder_model)
    for folder in ["models", "results"]:
        path = os.path.join(os.path.join(parent_dir, folder), folder_name)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as error:
            pass
    model_path = os.path.join(os.path.join(parent_dir, "models"), folder_name)
    result_path = os.path.join(os.path.join(parent_dir, "results"), folder_name)

    if encoder_model == "OTNAE":
        model = OTNAE(native_dim, hidden_layer)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_criterion = OTNAE_loss_criterion(native_dim)
        trainer = OTNAE_Trainer(native_dim, optimizer, model, loss_criterion)
    elif encoder_model == "AE":
        model = AE(native_dim, latent_dim, hidden_layer)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_criterion = torch.nn.MSELoss()
        trainer = AE_Trainer(native_dim, latent_dim, optimizer, model, loss_criterion)
    elif encoder_model == "VAE":
        model = VAE(native_dim, latent_dim, hidden_layer)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_criterion = VAE_loss_criterion()
        trainer = VAE_Trainer(native_dim, latent_dim, optimizer, model, loss_criterion, epochs)
    elif encoder_model == "OTNVAE":
        model = OTNVAE(native_dim, hidden_layer)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_criterion = OTNVAE_loss_criterion(native_dim)
        trainer = OTNVAE_Trainer(native_dim, optimizer, model, loss_criterion, epochs)

    for epoch in range(1, epochs+1):
        trainer.train(train_dataloader, epoch)
        trainer.test(train_dataloader, epoch, model_path)

    if encoder_model == "OTNAE":
        trainer.draw_records(title="Running Loss (env={}, encoder={})".format(env_id, encoder_model),result_path=result_path, num_records=args.num_records)
    elif encoder_model == "AE" or encoder_model == "AE":
        trainer.draw_records(title="Running Loss (env={}, encoder={})".format(env_id, encoder_model),result_path=result_path)
    elif encoder_model == "OTNVAE":
        trainer.draw_records(title="Running Loss (env={}, encoder={})".format(env_id, encoder_model),result_path=result_path, num_records=args.num_records)
    elif encoder_model == "VAE" or encoder_model == "VAE":
        trainer.draw_records(title="Running Loss (env={}, encoder={})".format(env_id, encoder_model),result_path=result_path)
    print(os.listdir())