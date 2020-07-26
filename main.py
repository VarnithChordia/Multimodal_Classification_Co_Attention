import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tensorboardX import SummaryWriter
from model import CoAttention
from dataloader import VQADataset, TextLoader
from dataloader_test import VQADatasetTest, TestLoader
from utils import int_min_two
from helpfunctions import accuracy, evaluation_metrics, save_checkpoint
from tqdm import tqdm
import numpy as np
import random


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def main():
    parser = argparse.ArgumentParser(description='Raukuten SIGIR competition')
    parser.add_argument('--mode',          type=str,            help='train or test mode', choices='train', default='train')
    parser.add_argument('--model',         type=str,            help='VQA model', choices=['bert'], default='bert')
    parser.add_argument('--model_name', type=str, help='model name', choices=['flaubert/flaubert_base_uncased','flaubert-base-cased','flaubert-large-cased','camembert/camembert-large', 'camembert-base', 'bert-base-multilingual-cased','xlm-roberta-base'])
    parser.add_argument('--model_type', type=str, help='model type', choices=['camem', 'flaubert', 'XLMRoberta', 'M-Bert'])
    parser.add_argument('--embs', type=int, help='Embedding dimension', default=768)
    parser.add_argument('--self_attention', type=bool, help='Self attention', default=False)
    parser.add_argument('--train_img',     type=str,            help='path to training images directory')
    parser.add_argument('--train_file',    type=str,            help='training dataset file')
    parser.add_argument('--labels_file',    type=str,           help='categories to label file')
    parser.add_argument('--val_img',       type=str,            help='path to validation images directory')
    parser.add_argument('--val_file',      type=str,            help='validation dataset file')
    parser.add_argument('--num_cls', '-K', type=int_min_two, help='Num of labels (labels)')
    parser.add_argument('--test_img', type=str, help='path to training images directory')
    parser.add_argument('--test_file', type=str, help='test dataset file')
    parser.add_argument('--batch_size',    '-bs',   type=int,   help='batch size')
    parser.add_argument('--num_epochs',    '-ep',   type=int,   help='number of epochs')
    parser.add_argument('--lr', '-lr', type=int, help='learning rate')
    parser.add_argument('--model_ckpt',    type=str,            help='resume training/perform inference; e.g. model_1000.pth')
    parser.add_argument('--num_workers',   type=int,            help='number of worker threads for Dataloader', default=1)
    parser.add_argument('--output_file', type=str, help='Store submission file')
    parser.add_argument('--output_model_name', type=str, help='Model name')
    parser.add_argument('--output_model_path', type=str, help='Model name path')
    parser.add_argument('--requires_grad', type=bool, help='requires gradient',default=True)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Selected Device: {}'.format(device))
    print('Available devices ', torch.cuda.device_count())
    # Train params
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    # Model Config
    model_config = setup_model_configs(args)
    image_size = model_config['image_size']

    # Train
    if args.mode == 'train':
        writer = SummaryWriter('')
        ##Transform
        transform = Compose([Resize(image_size), ToTensor(), Normalize((0.485, 0.456, 0.406),
                                                                       (0.229, 0.224, 0.225))])
        # Dataset & Dataloader
        train_dataset = VQADataset.create(image_dir=args.train_img, data_file=args.train_file,
                                          transform=transform, labels_path=args.labels_file,
                                          model_type=args.model_type, model_name=args.model_name)
        train_loader = TextLoader(train_dataset, device="cuda", batch_size = batch_size, num_workers=args.num_workers)


        if args.val_file:
            val_dataset = VQADataset.create(image_dir=args.train_img, data_file=args.val_file,
                                            transform=transform, labels_path=args.labels_file,
                                            model_type=args.model_type, model_name=args.model_name)

            val_loader = TextLoader(val_dataset, device="cuda", batch_size=batch_size, num_workers=args.num_workers)


        # Num of classes = K
        num_classes = args.num_cls

        # Setup model params
        question_encoder_params = model_config['question_params']
        image_encoder_params = model_config['image_params']

        # Define model & load to device
        PRDNet = model_config['model']

        model = PRDNet(question_encoder_params, image_encoder_params, K=num_classes)
        model = torch.nn.DataParallel(model)
        model.to(device)

        if args.model_ckpt:
            model_ckpt_path = args.model_ckpt
            checkpoint = torch.load(model_ckpt_path)
            print('The model has been loaded')
            model.load_state_dict(checkpoint)
            del checkpoint

        # Loss & Optimizer
        def configure_optimizers():
            params = list(model.named_parameters())
            return params

        def is_backbone(n):
            matches = ['image_encoder.resnet_encoder', 'question_encoder.model']
            if any(x in n for x in matches):
                 return True
            return False

        def require_gradients(parameter):
            if parameter.requires_grad:
                return True
            return False


        params = configure_optimizers()

        grouped_parameters = [
            {"params": [p for n, p in params if is_backbone(n) and require_gradients(p)], 'lr': args.lr * .1},
            {"params": [p for n, p in params if not is_backbone(n) and require_gradients(p)], 'lr': args.lr}
        ]

        optim = torch.optim.Adam(grouped_parameters, lr= args.lr, betas=(0.9, 0.999), eps=1e-08)

        criterion = nn.CrossEntropyLoss()

        decayRate = 0.9
        optimizer = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decayRate)

        best_accuracy_value = 0

        for epoch in range(n_epochs):
            train_loss = 0
            preds_train_cpu = []
            target_train_cpu = []
            for batch_data in tqdm(train_loader):

                image = batch_data['image'].to(device)
                input_ids = batch_data['input_id'].to(device)
                input_mask = batch_data['input_mask'].to(device)
                input_type_ids = batch_data['input_type_ids'].to(device)
                label = batch_data['labels'].to(device)
                # Forward Pass
                label_predict = model(image, input_ids, input_mask, input_type_ids)
                # Compute Loss
                loss = criterion(label_predict, label)
                train_loss += loss.item()
                #Calculate
                pred = label_predict.data.max(1, keepdim=True)[1]
                target = label.data.view_as(pred)
                preds_train_cpu.extend(pred.data.cpu())
                target_train_cpu.extend(target.data.cpu())
                # Backward Pass
                optim.zero_grad()
                loss.backward()
                optim.step()

            torch.cuda.empty_cache()

            if args.val_file:

                validation_metrics, validation_accuracy, val_loss = compute_validation_metrics(model, val_loader, device, criterion=criterion)
                accuracy_value_train = accuracy(preds_train_cpu, target_train_cpu)

                print('----------------------------------- Epoch : {}----------------------------------------'.format(
                    epoch))
                print(
                    'Total train_loss  : {:.2f}\n Total val loss : {:.2f}\n Train accuracy : {:.2f}\n Val Accuracy : {:.2f}\n'.format(
                        train_loss, val_loss, accuracy_value_train * 100, validation_accuracy * 100))
                print(validation_metrics[0])
                print(validation_metrics[1])

                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accu/train', accuracy_value_train, epoch)
                writer.add_scalar('Accu/val', validation_accuracy, epoch)

                print('-----------------------------------------------------------------------------------------------------')

                writer.flush()
                optimizer.step()


                if validation_accuracy > best_accuracy_value:
                    save_checkpoint({
                        "epoch": epoch + 1,
                        "model_name": '{}_{}_{}'.format(args.output_model_name, args.model_name, 'acc'),
                        "best_metrics_F1": validation_metrics[0],
                        "accuracy": validation_accuracy,
                        "valid_loss": val_loss,
                        "training_loss": train_loss
                    }, filename = os.path.join(args.output_model_path, '/{}-{}-{}.pth'.format(args.output_model_name, args.model_name, 'acc')),
                        model_name=model)
                    best_accuracy_value = validation_accuracy
                model.train()

    elif args.mode == 'test':
        question_encoder_params = model_config['question_params']
        image_encoder_params = model_config['image_params']
        PRDNet = model_config['model']
        transform = Compose([Resize(image_size), ToTensor(), Normalize((0.485, 0.456, 0.406),
                                                                       (0.229, 0.224, 0.225))])
        val_dataset = VQADatasetTest.create(image_dir=args.test_img, data_file=args.test_file,
                                            transform=transform, model_type=args.model_type,
                                            model_name=args.model_name)
        val_loader = TestLoader(val_dataset, device='cuda', batch_size=1)
        num_classes = args.num_cls
        model = PRDNet(question_encoder_params, image_encoder_params, K=num_classes)
        model = torch.nn.DataParallel(model)
        model.to(device)
        pred_labels = compute_test(args, model, val_loader, device)
        pd.DataFrame(pred_labels, columns=['Image_id', 'Prdtypecode']).to_csv(args.output_file, index=False)

def compute_validation_metrics(model, dataloader, device, criterion):
    """
    For the given model, computes accuracy & loss on validation/test set.

    :param model: VQA model
    :param dataloader: validation/test set dataloader
    :param device: cuda/cpu device where the model resides
    :param size: no. of samples (subset) to use
    :return: metrics {'accuracy', 'loss'}
    :rtype: dict
    """


    model.eval()
    with torch.no_grad():
        eval_loss = 0.0
        preds_cpu = list()
        target_cpu =list()

        # Evaluate on mini-batches
        for i, batch in tqdm(enumerate(dataloader)):
            # Load batch data
            image = batch['image']
            input_ids = batch['input_id']
            input_mask = batch['input_mask']
            input_type_ids = batch['input_type_ids']
            label = batch['labels']

            # Load data onto the available devicea
            image = image.to(device)  # [B, C, H, W]
            input_ids = input_ids.to(device)  # [B, L]
            input_mask = input_mask.to(device)  # [B, L]
            input_type_ids = input_type_ids.to(device)  # [B, L]
            label = label.to(device)  # [B]

            # Forward Pass
            label_logits = model(image, input_ids, input_mask, input_type_ids)

            pred = label_logits.data.max(1, keepdim=True)[1]
            target = label.data.view_as(pred)
            preds_cpu.extend(pred.data.cpu())
            target_cpu.extend(target.data.cpu())



            # Compute Loss
            val_loss = F.cross_entropy(label_logits, label)
            eval_loss += val_loss.item()
        classification_report = evaluation_metrics(preds_cpu, target_cpu)
        accuracy_value = accuracy(preds_cpu, target_cpu)
        return classification_report, accuracy_value, eval_loss



def compute_test(args, model, dataloader, device):
    """
    compute the values from test dataset
    :param args:
    :param model:
    :param dataloader:
    :param device:
    :return:
    """

    print('Beginning the testing phase')

    if args.model_ckpt:
        model_ckpt_path = args.model_ckpt
        checkpoint = torch.load(model_ckpt_path)
        model.load_state_dict(checkpoint)
        model.to(device)

    model.eval()
    with torch.no_grad():

        preds_cpu = list()

        # Evaluate on mini-batches
        for i, batch in tqdm(enumerate(dataloader)):
            # Load batch data
            image = batch['image']
            input_ids = batch['input_id']
            input_mask = batch['input_mask']
            input_type_ids = batch['input_type_ids']
            image_id = batch['image_id']

            # Load data onto the available device
            image = image.to(device)  # [B, C, H, W]
            input_ids = input_ids.to(device)  # [B, L]
            input_mask = input_mask.to(device)  # [B, L]
            input_type_ids = input_type_ids.to(device)  # [B, L]

            # Forward Pass
            label_logits = model(image, input_ids, input_mask, input_type_ids)

            ##Categories
            with open('categories_to_dicts.json','r') as f:
                idx2label = json.load(f)

            # Compute Accuracy
            pred = label_logits.data.max(1, keepdim=True)[1]
            final_pred = pred.data.cpu().numpy().ravel()
            image_id = image_id.numpy().ravel()
            preds_cpu.append((image_id[0], idx2label[str(final_pred[0])]))


        return  preds_cpu


def setup_model_configs(args):

    img_encoder_params = dict(is_require_grad=args.requires_grad)

    model_config = {'bert':dict(model=CoAttention,
                                image_size=(500, 500),
                                image_params=img_encoder_params,
                                question_params=dict(
                                    model_type=args.model_type,
                                    model_name=args.model_name,
                                    embedding_size=args.embs,
                                    hidden_dim=2048,
                                    rnn_layers=1,
                                    lstm_dropout=0.5,
                                    device="cuda",
                                    mode="weighted",
                                    self_attention= args.self_attention,
                                    is_require_grad=args.requires_grad),
                                    mlp_dim=1024,

                                )}

    return model_config[args.model]


if __name__ == '__main__':
    main()
