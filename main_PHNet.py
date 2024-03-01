import torch
import argparse
import csv
from CreateNiiDataset import CreateNiiDataset
from pytorch_metric_learning import distances, reducers, losses, miners
from net_hub.Densenet3D_embedding import densenet121
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score
import os
from model_utils.losses.losses import TripletCustomMarginLoss, LowerBoundLoss
from model_utils.miners.triplet_automargin_miner import TripletAutoParamsMiner
from model_utils.methods import MetricLearningMethods
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import hydra
import torchio as tio
from model_utils.get_logger import get_logger
from model_utils.calculate_auc_ci import calculate_auc_ci
from model_utils.LabelsmoothingBCE import LabelSmoothingLoss



def get_outpath(cfg):
    save_wight_path = os.path.join(cfg.save_path.path, cfg.save_path.weight)
    save_logger_path = os.path.join(cfg.save_path.path, cfg.save_path.logger)
    save_excle_path = os.path.join(cfg.save_path.path, cfg.save_path.excle)
    save_png_path = os.path.join(cfg.save_path.path, cfg.save_path.png)
    ouput_file_name = cfg.save_path.file_name
    return save_wight_path, save_logger_path, save_excle_path, save_png_path, ouput_file_name


def main():
    input_train_5fold_csv = os.path.join(cfg.csv_path, cfg.train_csv_path)
    input_val_5fold_csv = os.path.join(cfg.csv_path, cfg.val_csv_path)
    device = cfg.device
    Fold = args.Fold
    batch_size = cfg.batch_size
    model = cfg.model
    lr = cfg.lr
    epochs = cfg.epochs


    ######################################################################################
    with open(input_train_5fold_csv,'rt') as csvfile:
        reader = csv.reader(csvfile)
        for i,rows in enumerate(reader):
            if i == Fold:  # [i == 1,2,3,4,5 ====> 1,2,3,4,5 Fold]
                row_train = rows
                while '' in row_train:
                    row_train.remove('')
                row_train = row_train[1:]
    with open(input_val_5fold_csv,'rt') as csvfile:
        reader = csv.reader(csvfile)
        for i,rows in enumerate(reader):
            if i == Fold:  # [i == 1,2,3,4,5 ====> 1,2,3,4,5 Fold]
                row_val = rows
                while '' in row_val:
                    row_val.remove('')
                row_val = row_val[1:]
    print("open_csv done !")
    ######################################################################################
    train_data = CreateNiiDataset(row_train, row_val, train=True)
    val_data   = CreateNiiDataset(row_train, row_val, train=False)
    train_dataset = train_data.load_data()
    val_dataset   = val_data.load_data()
    train_num = len(train_dataset)
    val_num   = len(val_dataset)
    print("transform data done! ")
    ######################################################################################
    # data augument
    transform = tio.Compose([
        tio.RandomFlip(p=0.3),
        tio.RandomAffine(p=0.3),
        # tio.RandomElasticDeformation(num_control_points=10, locked_borders=True, p=0.2),
        # tio.HistogramStandardization(landmarks=landmarks),
        tio.RandomBiasField(p=0.3),
        tio.RandomNoise(p=0.3),
        tio.RandomGamma(p=0.3)
    ])
    train_dataset_torchio = tio.SubjectsDataset(train_dataset, transform=transform)
    val_dataset_torchio = tio.SubjectsDataset(val_dataset)
    print("improve data done!")
    ######################################################################################
    train_loader = torch.utils.data.DataLoader(train_dataset_torchio, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset_torchio, batch_size=batch_size, shuffle=False)
    print("preparing done !")
    ######################################################################################
    # net = eval(str(model))(args)
    net = densenet121(drop_rate=cfg.drop_rate)
    # net = cfg.model
    net.to(device)
    print("to net_hub done !")
    ######################################################################################
    if cfg.distance_loss == 'cosine':
        distance = distances.CosineSimilarity()
    elif cfg.distance_loss == 'l2':
        distance = distances.LpDistance()
    reducer = reducers.ThresholdReducer(low=0)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # next_lr = now_lr * 0.9
    # weight = torch.from_numpy(np.array([1.68, 2.46])).float().to(device)
    loss_function2 = LabelSmoothingLoss(num_classes=2, epsilon=cfg.smoothing)
    loss_function3 = nn.CrossEntropyLoss()
    loss_func = {'Triplet': TripletCustomMarginLoss(margin=cfg.margin.m_loss, distance=distance, reducer=reducer),
                 'LowerBoundLoss': LowerBoundLoss()}
    mining_func = {"AutoParams": TripletAutoParamsMiner(distance=distance, margin_init=cfg.margin.m_loss,
                                             beta_init=cfg.margin.beta,
                                             type_of_triplets=cfg.type_of_triplets,
                                             k=cfg.k_param_automargin, k_n=cfg.k_n_param_autobeta,
                                             k_p=cfg.k_p_param_autobeta,
                                             mode=cfg.automargin_mode)
    }
    list_an_dist, list_ap_dist = {}, {}
    best_acc , best_auc = 0.0, 0.0
    train_steps, val_steps = len(train_loader), len(val_loader)
    train_losses_list, val_loss_list, val_accurate_list, lr_list = [], [], [], []
    for epoch in range(epochs):
        if cfg.loss_identity_func == 'LSE':
            loss_id_selected = loss_func['LogSumExpLoss']
        elif cfg.loss_identity_func == 'LB':
            loss_id_selected = loss_func['LowerBoundLoss']
        else:
            raise ValueError(f'Not support this loss {cfg.loss_identity_func}')

        if epoch == 0:
            if cfg.method == 'AdaTriplet-AM':
                mining_func = mining_func['AutoParams']
                loss_matching_func = loss_func['Triplet']
                loss_id_func = loss_id_selected
            else:
                raise ValueError(f'Not support this method {cfg.method}')

        label_list, predict_list = [], []
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        sum_loss = 0
        sum_triplets = 0
        sum_neg_pairs = 0
        sum_pos_pairs = 0

        for batch_id, data in enumerate(train_bar):
            # train
            images = data['image']['data']
            labels = data['label']

            optimizer.zero_grad()
            out, out_embeddings = net(images.to(device))
            # choose adapt triplet loss
            method = MetricLearningMethods(cfg, mining_func, loss_matching=loss_matching_func,
                                           loss_identity=loss_id_func)

            # label smoothing
            smoothing = cfg.smoothing
            smooth_labels = (1 - smoothing) * labels + smoothing / 2
            loss1 = method.calculate_total_loss(out_embeddings.to("cpu"), smooth_labels, epoch_id=epoch, batch_id=batch_id)
            loss2 = loss_function2(out, labels.to(device))
            no_triplets_batch, no_neg_pairs_batch, no_pos_pairs_batch = method.get_no_triplets()
            if ~torch.isnan(loss1):
                sum_loss += loss1.to(device)
            sum_triplets += no_triplets_batch
            sum_neg_pairs += no_neg_pairs_batch
            sum_pos_pairs += no_pos_pairs_batch

            loss = cfg.adapt_loss_weight.weight1 * loss1 + cfg.adapt_loss_weight.weight2 * loss2
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            tripet_loss = loss1.item() * cfg.adapt_loss_weight.weight1
            BCE_loss = loss2.item() * cfg.adapt_loss_weight.weight2
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} BCEloss:{:.3f} adapt_loss:{:.5f} ".format(epoch + 1, epochs, train_loss, BCE_loss, tripet_loss)


        mean_loss = sum_loss / train_steps
        mean_triplets = sum_triplets / train_steps
        mean_neg_pairs = sum_neg_pairs / train_steps
        mean_pos_pairs = sum_pos_pairs / train_steps
        print(f'Average Loss: {mean_loss}')
        print(f'Average numbers of triplets: {mean_triplets}')
        print(f'Average numbers of negative pairs: {mean_neg_pairs}')
        print(f'Average numbers of positive pairs: {mean_pos_pairs}')


        # validate
        net.eval()
        acc = 0.0
        TP, FN, FP, TN = 0.0, 0.0, 0.0, 0.0
        val_loss = 0.0
        num_correct = 0.0
        with torch.no_grad():
            step_val = 0
            logger = get_logger(args.logger)  # training logger
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                step_val += 1
                val_images = val_data['image']['data']
                val_labels = val_data['label']
                outputs, outputs_embedding = net(val_images.to(device))

                val_loss = loss_function3(outputs, val_labels.to(device))
                val_loss += val_loss.item() * val_images.size(0)
                y_score = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.sum(torch.eq(predict_y, val_labels.to(device))).item()
                TP += torch.sum((val_labels.to(device) == 1) & (predict_y > 0.5)).item()
                FN += torch.sum((val_labels.to(device) == 1) & (predict_y < 0.5)).item()
                FP += torch.sum((val_labels.to(device) == 0) & (predict_y > 0.5)).item()
                TN += torch.sum((val_labels.to(device) == 0) & (predict_y < 0.5)).item()
                num_correct += torch.sum(predict_y > 0.5).item()
                label_list.append(val_labels.numpy())
                predict_list.append(y_score)
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        label_list = np.concatenate(label_list)
        predict_list = np.concatenate(predict_list)
        val_accurate = acc / val_num
        SE = TP / (TP + FN + 1e-7)  # avoid  0, + 1*10-7
        SP = TN / (TN + FP + 1e-7)
        PPV = TP / (TP + FP + 1e-7)
        NPV = TN / (TN + FN + 1e-7)
        F1_score = (2 * TP) / (2 * TP + FP + FN)
        train_losses = running_loss / train_steps
        val_loss = val_loss / val_steps
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_losses_list.append(train_losses)
        val_loss_list.append(val_loss.item())
        val_accurate_list.append(val_accurate)
        lr_list.append(lr)

        print('[epoch %d] train_loss: %.3f  val_loss: %.4f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps,  val_loss, val_accurate))

        logger.info(
            'train|epoch:{epoch}\tstep:{step}/{all_step}\ttrain_loss:{loss:.4f}\tval_loss:{val_loss:.4f}\tval_acc:{acc:.3f}\tSE:{SE:.3f}\tSP:{SP:.3f}\tPPV:{PPV:.3f}\tNPV:{NPV:.3f}\tF1_score:{F1_score:.3f}\tlr:{lr:.2e}'.format(
                epoch=epoch, step=step_val + 1,
                all_step=len(val_loader), loss=train_losses,val_loss=val_loss, acc=float(val_accurate), SE=SE, SP=SP,
                PPV=PPV, NPV=NPV, F1_score=F1_score, lr=lr))

        # write ROC、AUC
        if epoch > 10:
            fpr, tpr, thersholds = roc_curve(label_list, predict_list)
            roc_auc = auc(fpr, tpr)
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_acc = val_accurate
                torch.save(net.state_dict(), args.save_path)
                print(predict_list)

                bootstrapped_auc, lower_bound, upper_bound = calculate_auc_ci(label_list, predict_list)

                roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thersholds})
                wb = Workbook()
                ws = wb.active
                ws.title = "ROC"
                for r in dataframe_to_rows(roc_df, index=False, header=True):
                    ws.append(r)

                auc_interval_df = pd.DataFrame(
                    {"AUC": [auc], "lower Bound": [lower_bound], "Upper Bound": [upper_bound]}, index=[0])
                auc_interval_df.to_csv(args.excle, index=False)
                wb.save(args.excle)

                plt.figure()
                lw = 2
                plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label="ROC curve (area = %0.3f)" % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.fill_between(fpr, tpr, color='gray', alpha=0.3,
                                 label='Confidence Interval: [{:.3f}, {:.3f}]'.format(lower_bound, upper_bound))
                plt.xlim([-0.01, 1.0])
                plt.ylim([-0.01, 1.0])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.xticks(np.arange(0, 1.1, 0.1))
                plt.yticks(np.arange(0, 1.1, 0.1))
                plt.savefig(os.path.join(args.savefig, "{}_ROC_AUC.png".format(args.name)))
    ######################################################################################
    # write losses

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(np.arange(len(train_losses_list)), val_accurate_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(np.arange(len(train_losses_list)), train_losses_list, '--', color=color, label='Train Loss')
    ax2.plot(np.arange(len(train_losses_list)), val_loss_list, '-', color=color, label='Val Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.savefig, "{}_loss.png".format(args.name)))

    print('best acc: best_auc', best_acc, best_auc)
    print('Finished Training')


if __name__ == '__main__':
    with hydra.initialize_config_dir(config_dir=os.getcwd()):
        cfg = hydra.compose(config_name="config.yaml")
    save_wight_path, save_logger_path, save_excle_path, save_png_path, ouput_file_name = get_outpath(cfg)
    for fold in range(5):
        fold_num = fold + 1
        parser = argparse.ArgumentParser()
        parser.add_argument("--Fold", type=int, default=fold_num, help="Fold == 1,2,3,4,5 ====> 0,1,2,3,4Fold")
        parser.add_argument("--save_path", type=str,
                            default=os.path.join(save_wight_path, ouput_file_name, "Fold{}.pth".format(fold_num)))
        parser.add_argument("--logger", type=str, default=os.path.join(save_logger_path, ouput_file_name, "Fold{}.txt".format(fold_num)))
        parser.add_argument("--savefig", type=str, default=os.path.join(save_png_path, ouput_file_name))
        parser.add_argument("--excle", type=str, default=os.path.join(save_excle_path, ouput_file_name, "Fold{}.csv".format(fold_num)))
        parser.add_argument("--name", type=str, default=r'Fold{}'.format(fold_num))
        args = parser.parse_args()
        main()
