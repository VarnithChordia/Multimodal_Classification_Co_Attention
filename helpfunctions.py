from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
import torch
import json
import os

def accuracy(output,target):
    '''
    :param output:
    :param target:
    :return:
    '''
    return accuracy_score(output, target)

def evaluation_metrics(preds, target) -> tuple:
    eval_metrics = classification_report(preds, target)
    precision = precision_score(preds, target, average='macro')
    recall = recall_score(preds, target, average='macro')
    F1_score = f1_score(preds, target, average='macro')
    return ({'precision':precision, 'recall':recall, 'F1_score':F1_score}, eval_metrics)


def save_checkpoint(config, filename, model_name, args):
    model = model_name
    torch.save(model.state_dict(), filename)
    with open(os.path.join(args.model_path, '{}.txt'.format(config['model_name'])), 'w') as f:
        f.write(json.dumps(config))


