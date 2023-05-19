import argparse
import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from pdbl import PDBLNet
from dataset import PDBLDataSet
from models import resnet50, shufflenet_v2_x1_0, EfficientNet
from sklearn.metrics import accuracy_score, f1_score

class Solver :
    def __init__(self, args):
        self.args = args

    def create_model(self, model_name, n_classes):
        if model_name == 'shufflenet' :
            model = shufflenet_v2_x1_0(pretrained = True)
            model.fc = nn.Linear(model._stage_out_channels[-1], n_classes)
        elif model_name == 'efficientnet' :
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes = n_classes)
        elif model_name == 'resnet' :
            model = resnet50(pretrained = True)
            model.fc = nn.Linear(512 * model.block.expansion, n_classes)
        else :
            raise ValueError('model should be shufflenet or efficientnet or resnet!')

        return model

    def save_PDBL(self, PDBL, fname):
        with open(Path(self.args.save_dir).joinpath(fname).__str__(), 'wb') as fp :
            pickle.dump(PDBL, fp)

    def load_PDBL(self, fname):
        with open(Path(self.args.save_dir).joinpath(fname).__str__(), 'rb') as fp :
            PDBL = pickle.load(fp)

        return PDBL

    def save_model(self, model, fname):
        torch.save(model.state_dict(), Path(self.args.save_dir).joinpath(fname).__str__())

    def load_model(self, model, fname):
        ckpt = torch.load(Path(self.args.save_dir).joinpath(fname).__str__(), map_location = self.args.device)
        model.load_state_dict(ckpt)

        return model

    def train_model(self, model, train_loader):
        if self.args.train_fc :
            model.eval()
        else :
            model.train()
        if self.args.train_fc :
            for param in list(model.parameters())[ : -2]:
                param.requires_grad = False

        # total_params = sum(p.numel() for p in model.parameters())
        # print('原总参数个数:{}'.format(total_params))
        # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print('需训练参数个数:{}'.format(total_trainable_params))

        optimizer = optim.Adam(model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        losses, acces = [], []

        for epoch in range(self.args.max_epoch) :
            train_loss = 0.0
            train_acc = 0.0
            for img, label in train_loader:
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)

                img, label = img.to(self.args.device), label.to(self.args.device)
                _, output = model(img)

                _, pred = torch.max(output, dim = 1)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_correct = (pred == label).sum().item()
                acc = num_correct / len(label)
                train_loss += loss.item()
                train_acc += acc

            # print(f'[{epoch + 1:02d}/{self.args.max_epoch:02d}], train_loss:{train_loss / len(train_loader):.5f}, '
            #       f'train_acc:{train_acc / len(train_loader):.5f}')

            losses.append(round(train_loss / len(train_loader), 5))
            acces.append(round(train_acc / len(train_loader), 5))
        # #
        # print('loss')
        # for loss in losses :
        #     print(loss)
        # print('acc')
        # for acc in acces :
        #     print(acc)

        return model

    def valid_model(self, model, val_loader):
        model.eval()
        model_true = []
        model_pred = []
        with torch.no_grad() :
            for img, label in val_loader:
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)

                img, label = img.to(self.args.device), label.to(self.args.device)
                _, output = model(img)
                _, pred = torch.max(output, dim = 1)

                num_correct = (pred == label).sum().item()
                acc = num_correct / len(label)

                model_true.extend(label.detach().cpu().numpy())
                model_pred.extend(pred.detach().cpu().numpy())

        model_acc = accuracy_score(model_true, model_pred)
        model_f1 = f1_score(model_true, model_pred, average = 'macro')

        print(f'Accuracy of {self.args.model_name}: ', model_acc)
        print(f'F1 score of {self.args.model_name}: ', model_f1)

    def train_PDBL(self, model, train_loader, PDBL):
        model.eval()
        work_space_in = np.zeros((self.args.n_item_train, self.args.n_features))
        work_space_out = np.zeros((self.args.n_item_train, self.args.n_classes))
        progress = 0

        for img224, img160, img112, label in tqdm(train_loader):
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            len_batch = len(label)

            with torch.no_grad() :
                img224 = Variable(img224.float().to(self.args.device))
                img160 = Variable(img160.float().to(self.args.device))
                img112 = Variable(img112.float().to(self.args.device))

            feature1, _ = model(img224)
            feature2, _ = model(img160)
            feature3, _ = model(img112)

            feature = feature1
            feature = torch.cat((feature, feature2), dim = 1)
            feature = torch.cat((feature, feature3), dim = 1)

            work_space_in[progress : progress + len_batch, : ] = feature.detach().cpu().numpy()
            work_space_out[progress : progress + len_batch , : ] = label.detach().cpu().numpy()

            progress += len_batch

        PDBL.fit(work_space_in, work_space_out)
        return PDBL

    def valid_PDBL(self, model, val_loader, PDBL):
        model.eval()

        work_space_in = np.zeros((self.args.n_item_val, self.args.n_features))
        work_space_out = np.zeros((self.args.n_item_val, self.args.n_classes))
        progress = 0

        for img224, img160, img112, label in tqdm(val_loader):
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            len_batch = len(label)

            with torch.no_grad():
                img224 = Variable(img224.float().to(self.args.device))
                img160 = Variable(img160.float().to(self.args.device))
                img112 = Variable(img112.float().to(self.args.device))

            feature1, _ = model(img224)
            feature2, _ = model(img160)
            feature3, _ = model(img112)

            feature = feature1
            feature = torch.cat((feature, feature2), dim = 1)
            feature = torch.cat((feature, feature3), dim = 1)

            work_space_in[progress: progress + len_batch, :] = feature.detach().cpu().numpy()
            work_space_out[progress: progress + len_batch, :] = label.detach().cpu().numpy()

            progress += len_batch

        PDBL_output = PDBL.predict(work_space_in)
        PDBL_pred = np.zeros(PDBL_output.shape[0])
        PDBL_true = np.zeros(PDBL_output.shape[0])

        for i in range(len(PDBL_output)) :
            PDBL_pred[i] = np.argmax(PDBL_output[i])
            PDBL_true[i] = np.argmax(work_space_out[i])

        PDBL_acc = accuracy_score(PDBL_true, PDBL_pred)
        PDBL_f1 = f1_score(PDBL_true, PDBL_pred, average='macro')

        print('Accuracy of PDBL: ', PDBL_acc)
        print('F1 score of PDBL: ', PDBL_f1)

    def run(self) :
        train_dataset = PDBLDataSet(data_path = self.args.train_dir, n_classes = self.args.n_classes, train_model = self.args.train_model)
        val_dataset = PDBLDataSet(data_path = self.args.val_dir, n_classes = self.args.n_classes, train_model = self.args.train_model)
        train_loader = DataLoader(train_dataset, batch_size = self.args.batch_size, shuffle = self.args.train_model)
        val_loader = DataLoader(val_dataset, batch_size = self.args.batch_size, shuffle = False)

        self.args.n_item_train = len(train_dataset)
        self.args.n_item_val = len(val_dataset)

        models_name = ['shufflenet', 'efficientnet', 'resnet']
        n_features = [812 * 3, 1456 * 3, 3840 * 3]

        for idx in range(len(models_name)) :
            self.args.model_name = models_name[idx]
            self.args.n_features = n_features[idx]
            print('<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('This model is', self.args.model_name)

            scale = 0.001
            n_components = min(int(self.args.n_item_train * 0.9), self.args.n_components)
            PDBL = PDBLNet(isPCA = True, n_components = n_components, scale = scale)
            model = self.create_model(model_name = self.args.model_name, n_classes = self.args.n_classes)
            model = model.to(self.args.device)

            if not Path(self.args.save_dir).exists() :
                Path(self.args.save_dir).mkdir(exist_ok = True, parents = True)

            if self.args.train_model :
                model = self.train_model(model, train_loader)
                self.valid_model(model, val_loader)
                if self.args.train_fc :
                    fname = f'train_fc_on_{models_name[idx]}.pth'
                else :
                    fname = f'fine_tuning_on_{models_name[idx]}.pth'
                self.save_model(model, fname)
                # model = self.load_model(model, fname)
                # self.valid_model(model, val_loader)
            else :
                if self.args.fine_tuning:
                    fname = f'fine_tuning_on_{models_name[idx]}.pth'
                    model = self.load_model(model, fname)

                PDBL = self.train_PDBL(model, train_loader, PDBL)
                self.valid_PDBL(model, val_loader, PDBL)

                if self.args.fine_tuning :
                    fname = f'PDBL_with_finetune_on_{models_name[idx]}.pkl'
                else :
                    fname = f'PDBL_on_{models_name[idx]}.pkl'
                self.save_PDBL(PDBL, fname)

                # PDBL_reload = self.load_PDBL(fname)

# if __name__ == '__main__':
#     n_components = [4000]
#     path = 'PDBL_Dataset/LC_Train_001'
#     for d in n_components :
#         print(f'<<<<<<<<<<<<<<<n_components : {d}>>>>>>>>>>>>>>>')
#         train_model = [True, False, True, False]
#         train_fc = [True, False, False, False]
#         fine_tuning = [False, False, True, True]
#         train_dir = [['PDBL_Dataset/Kather/KMI_001', 'PDBL_Dataset/Kather/KMI_001', 'PDBL_Dataset/Kather/KMI_001', 'PDBL_Dataset/Kather/KMI_001'],
#                      ['PDBL_Dataset/LC_Train_001', 'PDBL_Dataset/LC_Train_001', 'PDBL_Dataset/LC_Train_001', 'PDBL_Dataset/LC_Train_001']]
#         val_dir = [['PDBL_Dataset/Kather/CRC-VAL-HE-7K', 'PDBL_Dataset/Kather/CRC-VAL-HE-7K', 'PDBL_Dataset/Kather/CRC-VAL-HE-7K', 'PDBL_Dataset/Kather/CRC-VAL-HE-7K'],
#                    ['PDBL_Dataset/LC_Test', 'PDBL_Dataset/LC_Test', 'PDBL_Dataset/LC_Test', 'PDBL_Dataset/LC_Test']]
#         classes = [9, 5]
#         dataset = ['Kather', 'LC25000']
#         four_models = [f'Baseline+FC*', f'Baseline+PDBL', f'Baseline*', f'Baseline*+PDBL']
#
#         save_paths = [f'checkpoint/KMI_001_d_{d}', f'checkpoint/LC_001_d_{d}']
#         for i in range(2) :
#             print(f'<<<<<<<<<<<<<<<dataset : {dataset[i]}>>>>>>>>>>>>>>>')
#             for j in range(4) :
#                 print(f'<<<<<<<<<<<<<<<model : {four_models[j]}>>>>>>>>>>>>>>>')
#                 parser = argparse.ArgumentParser()
#                 parser.add_argument('--device', type = str, default = "cuda:1" if torch.cuda.is_available() else "cpu")
#                 parser.add_argument('--save_dir', type = str, default = save_paths[i])
#                 parser.add_argument('--train_dir', type = str, default = train_dir[i][j])
#                 parser.add_argument('--val_dir', type = str, default = val_dir[i][j])
#                 parser.add_argument('--n_classes', type=int, default = classes[i])
#                 parser.add_argument('--batch_size', type = int, default = 20)
#                 parser.add_argument('--max_epoch', type = int, default = 50)
#                 parser.add_argument('--lr', type = float, default = 1e-3)
#                 parser.add_argument('--weight_decay', type = float, default = 1e-4)
#                 parser.add_argument('--train_model', type = bool, default = train_model[j])
#                 parser.add_argument('--train_fc', type = bool, default = train_fc[j])
#                 parser.add_argument('--fine_tuning', type = bool, default = fine_tuning[j])
#                 parser.add_argument('--n_components', type = int, default = d)
#                 args = parser.parse_args()
#
#                 solver = Solver(args)
#                 solver.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type = str, default = "cuda:8" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--save_dir', type = str, default = 'checkpoint/Kather_001')
    parser.add_argument('--train_dir', type = str, default = 'PDBL_Dataset/Kather/KMI_001')
    parser.add_argument('--val_dir', type = str, default = 'PDBL_Dataset/Kather/CRC-VAL-HE-7K')
    parser.add_argument('--n_classes', type=int, default = 9)
    parser.add_argument('--batch_size', type = int, default = 20)
    parser.add_argument('--max_epoch', type = int, default = 50)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--weight_decay', type = float, default = 1e-4)
    parser.add_argument('--train_model', type = bool, default = False)
    parser.add_argument('--train_fc', type = bool, default = False)
    parser.add_argument('--fine_tuning', type = bool, default = False)
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()