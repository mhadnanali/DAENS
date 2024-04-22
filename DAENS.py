import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, CitationFull, Coauthor, Actor
import random
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, jaccard_score, f1_score
import pandas as pd
import time
import datetime
from sklearn.manifold import TSNE
import subprocess
import json
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from HyperParameters import HyperParameters
from torch_geometric.utils import dropout_adj
from typing import Optional, Tuple, NamedTuple, List
torch.cuda.empty_cache()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch_geometric.loader import ClusterData, ClusterLoader


def get_gpu_memory_usage():
    cmd = "gpustat --json"
    result = subprocess.check_output(cmd.split()).decode("utf-8")
    gpu_info = json.loads(result)["gpus"]
    gpu_memory_usage = []
    for gpu in gpu_info:
        for process in gpu["processes"]:
            gpu_memory_usage.append({"GPU": gpu["index"], "PID": process["pid"], "User": process["username"],
                                     "Memory": process["gpu_memory_usage"]})
    return gpu_memory_usage


def memory_calculate(memory_df):
    # Get GPU memory usage and add it to the DataFrame
    gpu_memory_usage = get_gpu_memory_usage()
    current_time = time.time()
    human_readable_time = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
    memory_data = {"Time": human_readable_time}  # Convert timestamp to human-readable time
    for gpu_usage in gpu_memory_usage:
        gpu = gpu_usage["GPU"]
        pid = gpu_usage["PID"]
        username = gpu_usage["User"]
        memory = gpu_usage["Memory"]
        column_name = f"GPU {gpu} (PID {pid}) - User: {username}"
        memory_data[column_name] = memory
    memory_df = pd.concat([memory_df, pd.DataFrame(memory_data, index=[0])], ignore_index=True)
    return memory_df


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class GaTeConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GaTeConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def getMasks(anchor, sample, data, ad):
    assert anchor.size(0) == sample.size(0)
    num_nodes = anchor.size(0)  # getting the number of nodes
    device = anchor.device
    pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
    neg_mask = 1. - pos_mask
    neg_mask.to(device)
    return pos_mask, neg_mask



def train(ad, encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    pos_mask1, neg_mask1 = getMasks(h1, h2, data, ad)
    pos_mask2, neg_mask2 = getMasks(h2, h1, data, ad)
    l1 = InfoNCE.compute(0, h1, h2, pos_mask1, neg_mask1, ad[2]['tau'])
    l2 = InfoNCE.compute(0, h2, h1, pos_mask2, neg_mask2, ad[2]['tau'])
    loss = (l1 + l2) * 0.5
    loss.backward()
    optimizer.step()
    return loss.item()


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split)
        return result


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')
                    t_ACCURACY = accuracy_score(y_test, y_pred)
                    t_JaccardScore = jaccard_score(y_test, y_pred, pos_label=1, average='micro')
                    t_RECALL = recall_score(y_test, y_pred, average='micro')
                    t_PRECISION = precision_score(y_test, y_pred, average='micro')
                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')
                    classes = np.unique(y_test)  
                    y_test_binarized = label_binarize(y_test, classes=classes)
                    y_pred_binarized = label_binarize(y_pred, classes=classes)
                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch
                        best_t_ACCURACY = t_ACCURACY
                        best_t_JaccardScore = t_JaccardScore
                        best_t_RECALL = t_RECALL
                        best_t_PRECISION = t_PRECISION
                        
                        lists = [best_t_ACCURACY, best_t_JaccardScore, best_t_RECALL, best_t_PRECISION,best_t_auc]

                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'otherscores': lists
        }


def test(encoder_model, data, dataset, D, epoch):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    print(data.x.device)
    if D["TSNE"] == "Yes":
        ###To create figures
        tsne = TSNE(n_components=2)
        z_tsne = tsne.fit_transform(z.cpu().detach().numpy())
        fig, ax = plt.subplots()
        data_y_cpu = data.y.cpu()
        for i in range(dataset.num_classes):
            ax.scatter(z_tsne[data_y_cpu == i, 0], z_tsne[data_y_cpu == i, 1], label=str(i))
        ax.legend()
        plt.title(f't-SNE at: (Epoch {epoch}, F1-micro: {result["micro_f1"]:.4f})')  # visualization of GCN embeddings
        # plt.xlabel('t-SNE feature 1')
        # plt.ylabel('t-SNE feature 2')
        folder_path = create_folder(D['Dataset'] + ' tSNE PDF')
        pdf_filename = f'/t-SNE_epoch_{epoch}_F1micro_{result["micro_f1"]:.2f}.pdf'
        with PdfPages(folder_path + pdf_filename) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
    else:
        pass
    return result


class Sampler(ABC):
    def __init__(self, adj_matrix, intraview_negs=False):
        self.intraview_negs = intraview_negs
        self.adj_matrix = adj_matrix

    def __call__(self, adj_matrix, anchor, sample, *args, **kwargs):
        ret = self.sample(adj_matrix, anchor, sample, *args, **kwargs)
        if self.intraview_negs:
            ret = self.add_intraview_negs(*ret)  # if intraview is true then add mask accordingly
        return ret

    @abstractmethod
    def sample(self, adj_matrix, anchor, sample, *args, **kwargs):
        pass

    @staticmethod
    def add_intraview_negs(anchor, sample, pos_mask, neg_mask):
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, device=device)
        intraview_neg_mask = torch.ones_like(pos_mask, device=device) - torch.eye(num_nodes, device=device)
        new_sample = torch.cat([sample, anchor], dim=0)  # (M+N) * K
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)  # M * (M+N)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)  # M * (M+N)
        return anchor, new_sample, new_pos_mask, new_neg_mask


class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss


class SameScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleSampler, self).__init__(*args, **kwargs)
        # self.data=data

    # this method decide the negative and postive nodes mask...
    def sample(self, adj_matrix, anchor, sample, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)  # getting the number of nodes
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask



def get_sampler(adj_matrix, mode: str, intraview_negs: bool) -> Sampler:
    if mode in {'L2L'}:
        return SameScaleSampler(adj_matrix, intraview_negs=intraview_negs)


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, tau):
        sim = _similarity(anchor, sample) / tau  # _similarity() normalize the tensors as it is must do for cosine similarity.
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))  # log compresses the large  values
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()





class DualBranchContrast(torch.nn.Module):
    def __init__(self, adj_matrix, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(adj_matrix, mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def add_extra_mask(self, pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
        if extra_pos_mask is not None:
            pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
        if extra_neg_mask is not None:
            neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
        else:
            neg_mask = 1. - pos_mask
        return pos_mask, neg_mask

    def forward(self, adj_matrix, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor, sample, pos_mask, neg_mask = self.sampler(adj_matrix, anchor=h1, sample=h2)
        pos_mask, neg_mask = self.add_extra_mask(pos_mask, neg_mask, extra_pos_mask, extra_neg_mask)
        return anchor, sample, pos_mask, neg_mask





class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class EdgeRemoving():
    def __init__(self, pe: float):
        super(EdgeRemoving, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x


class FeatureMasking():
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class ImportanceFeatures():
    def __init__(self, data, drop_prob, device, percentage):
        super(ImportanceFeatures, self).__init__()
        self.edge_index_ = None
        self.device = device

        self.drop_prob = drop_prob
        self.percentage = percentage
        self.data = data
        self.data.to(self.device)

        self.topfeatures = self.TopFeatureFinder(self.data, self.drop_prob, self.percentage)

    def augment(self, g: Graph) -> Graph:
        x, self.edge_index_, edge_weights = g.unfold()
        indices = torch.tensor(self.topfeatures, device=self.device)
        candidateCF = torch.index_select(x, 1, indices)
        XT = self.DropFeatures(x, candidateCF, indices, self.drop_prob, self.device, self.topfeatures)
        return Graph(x=XT, edge_index=self.edge_index_, edge_weights=edge_weights)

    def TopFeatureFinder(self, data, drop_prob, percentage):
        print("Random Feature finder called")
        totalFeatures = data.x.size()[1]
        totalRows = data.x.size()[0]
        random.seed(14112)
        topfeatures = list(range(0, totalFeatures))
        random.shuffle(topfeatures)
        tenper = (totalFeatures * percentage)
        topfeatures = topfeatures[:int(tenper)]
        print(topfeatures)
        return topfeatures

    def DropFeatures(self, X, candidateCF, indices, drop_prob, device, topfeatures):
        mask = torch.empty(candidateCF.size(), device=device).uniform_() > drop_prob
        output = candidateCF.mul(mask)
        X = X.clone()
        X[:, indices] = output  # .bool().float()

        return X


class Augmentor(ABC):
    """Base class for graph augmentors."""

    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()


class Compose(Augmentor):
    def __init__(self, augmentors: List[Augmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


def create_folder(folder_name):
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")
    return folder_path


def WithBatches(D, path, dsName, device, dataset, d):
    data = dataset[0]  # .to(device)
    dataloadersList = []
    filenamesList = []
    if D["batches"] == "yes":
        print("Creating batches")
        torch.manual_seed(11314)
        cluster_data = ClusterData(data, num_parts=D['cluster_num_parts'])  
        train_loader = ClusterLoader(cluster_data, batch_size=D['cluster_batch_size'],shuffle=True)  
        total_num_nodes = 0
        for step, sub_data in enumerate(train_loader):
            adjmatrix_file = str(step) + dsName + " " + str(D["cluster_num_parts"]) + "x" + str(
                D["cluster_batch_size"]) + " Adjacency matrix.csv"
            adjm_filename = path + "/" + adjmatrix_file
            filenamesList.append(adjm_filename)
            print(f'Step {step + 1}:')
            print('=======')
            print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
            print(sub_data)
            dataloadersList.append(sub_data)
            print()
            total_num_nodes += sub_data.num_nodes
        print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')
        print(filenamesList)
    elif D["batches"] == "no":
        adjmatrix_file = str("NoBatch") + dsName + " Adjacency matrix.csv"
        adjm_filename = path + "/" + adjmatrix_file
        filenamesList.append(adjm_filename)
        dataloadersList.append(data)
    if len(dataloadersList) > 1:
        print("Cluster Created")
    else:
        print("No Clusters")

    drop_prob = D['Aug2_FM']
    percentage = D['percentage']
    pf = str(drop_prob) + 'x' + str(percentage)  # drop_prob x Percentage
    outputFile = dsName + " " + D['Base_Algo'] + " " + pf + " " + ".csv"
    mdf = outputFile.split('.')
    memory_df_file = mdf[0] + " " + pf + " Memory.csv"
    print("memory filename is:", memory_df_file)
    memory_df = pd.DataFrame(columns=["Time"])
    if os.path.isfile(outputFile):
        df = pd.read_csv(outputFile, index_col=0)
    else:
        df = pd.DataFrame(columns=['Algo', 'Dataset', 'TotalEpochs', 'TotalDrop', 'MicroF1', 'MacroF1','Accuracy', 'AUC','Recall', 'Precision', 'JS',
                                   'AugType', "NegSamples", "Epoch", "Result", "Seeds", "Time", "Batches",
                                   "hyper", "Tau", "Encoder"])

    data = data.to(device)
    for xws in range(0, D["rounds"]):
        print("Round: ", xws)
        if xws > 0:
            D["TSNE"] = "No"
        Oneseed = D['training_manual_seed']
        twoseed = D['training_random_seed']
        torch.manual_seed(Oneseed)
        random.seed(twoseed)
        import time
        start_time = time.perf_counter()
        if D['Type'] == "1D2D":

            aug1 = Compose([EdgeRemoving(pe=D['Aug1_ER']), FeatureMasking(pf=D['Aug1_FM'])])  # sending as list
            aug2 = Compose([EdgeRemoving(pe=D['Aug2_ER']), ImportanceFeatures(data, drop_prob, device, percentage)])
        elif D['Type'] == "Both1D":
            aug1 = Compose([EdgeRemoving(pe=D['Aug1_ER']), FeatureMasking(pf=D['Aug1_FM'])])
            aug2 = Compose([EdgeRemoving(pe=D['Aug2_ER']), FeatureMasking(pf=D['Aug2_FM'])])
        elif D['Type'] == "Both2D":
            aug1 = Compose([EdgeRemoving(pe=D['Aug1_ER']), ImportanceFeatures(data, D['Aug1_FM'], device, 1)])
            aug2 = Compose([EdgeRemoving(pe=D['Aug2_ER']), ImportanceFeatures(data, D['Aug2_FM'], device, percentage)])
        adj_matrix = None

        if D["Encoder"] == "GCN":
            gconv = GConv(input_dim=dataset.num_features, hidden_dim=D['hidden_dim'], activation=D['activation'],
                          num_layers=D['num_layers']).to(device)
        elif D["Encoder"] == "GAT":
            gconv = GaTeConv(input_dim=dataset.num_features, hidden_dim=D['hidden_dim'], activation=D['activation'],
                             num_layers=D['num_layers']).to(device)
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=D['hidden_dim'],
                                proj_dim=D['proj_dim']).to(device)

        contrast_model = DualBranchContrast(adj_matrix, loss=InfoNCE(tau=D['tau']), mode='L2L',
                                            intraview_negs=D['intraview_negs']).to(device)
        optimizer = Adam(encoder_model.parameters(), lr=D['lr'])

        epochs = D['train_epochs']
        r = []
        r2 = []
        r3_acc = []
        r3_JS = []
        r3_RE = []
        r3_PR = []
        r3_auc = []
        highestResults = (0, 0)
        losslist = []
        ad = (adj_matrix, adjmatrix_file, D)
        highestResults = (0, 0)
        if len(dataloadersList) > 1:
            tqdmEpoch = epochs * len(dataloadersList)
        else:
            tqdmEpoch = epochs
        with tqdm(total=tqdmEpoch, desc='(T)') as pbar:
            for epoch in range(1, epochs):

                for sub_data in dataloadersList:
                    sub_data = sub_data.to(device)
                    loss = train(ad, encoder_model, contrast_model, sub_data, optimizer)
                    pbar.set_postfix({'loss': loss})
                    pbar.update()
                if epoch % D['test_epochs'] == 0:
                    test_result = test(encoder_model, data, dataset, D, epoch)
                    r.append(test_result["micro_f1"])
                    r2.append(test_result["macro_f1"])
                    r3_acc.append(test_result['otherscores'][0])  
                    r3_JS.append(test_result['otherscores'][1])  
                    r3_RE.append(test_result['otherscores'][2])  
                    r3_PR.append(test_result['otherscores'][3])  
                    r3_auc.append(test_result['otherscores'][4]) 



                    if highestResults[1] < test_result["micro_f1"]:
                        highestResults = (epoch, test_result["micro_f1"])
                    if d != "cpu" and epoch == (epochs - 1):
                        memory_df = memory_calculate(memory_df)

        max_value = round(max(r), 4)
        max_value2 = round(max(r2), 4)
        max_value3 = round(max(r3_acc), 4)
        max_value4 = round(max(r3_JS), 4)
        max_value5 = round(max(r3_RE), 4)
        max_value6 = round(max(r3_PR), 4)

        print(f'(E): Best test F1Mi={max_value:.4f}, F1Ma={max_value2:.4f}')
        end_time = time.perf_counter()
        values_to_add = {'Algo': D['Base_Algo'], 'Dataset': dsName, 'TotalEpochs': epochs - 1, 'TotalDrop': pf,
                         'MicroF1': max_value, 'MacroF1': max_value2, 'Accuracy': max_value3,  
                         'Recall': max_value5, 'Precision':max_value6 , 'JS':max_value4 ,
                         'AugType': str(D['Type']),
                         "NegSamples": str(D['intraview_negs']),

                         "Epoch": highestResults[0], "Result": highestResults[1],
                         'Seeds': str(D['main_manual_seed']) + " " + str(D['main_random_seed']) + " " + str(
                             D['training_manual_seed']) + " " + str(D['training_random_seed']),
                         "Time": end_time - start_time, "Batches": len(dataloadersList),
                         "hyper": str(D['Aug1_ER']) + " " + str(D['Aug1_FM']) + " " + str(D['Aug2_ER']) + " " + str(
                             D['Aug2_FM']) + " " + str(percentage), "Tau": D['tau'], "Encoder": D["Encoder"]
                         }

        df.loc[len(df)] = pd.Series(values_to_add)
        memory_df.to_csv(memory_df_file, index=False)
        df.to_csv(outputFile)


def main():
    d = "cuda:2"
    #d = "cpu"
    dsName = 'CS' #Enter Dataset Name Here
    hyper = HyperParameters()
    D = hyper.datasetName(dsName)
    print(D)
    torch.manual_seed(D['main_manual_seed'])
    random.seed(D['main_random_seed'])
    device = torch.device(d)

    path = osp.join(osp.expanduser('~'), D['dataset_P_folder'], D['dataset_C_folder'])
    if dsName == "CiteSeer" or dsName == "Cora" or dsName == "PubMed":
        dataset = Planetoid(path, name=dsName)
    elif dsName == "dblp":
        dataset = CitationFull(path, name=dsName)
    elif dsName == "WikiCS":
        dataset = WikiCS(path)
    elif dsName == "photo" or dsName == "computers":
        dataset = Amazon(path, name=dsName)
    elif dsName == "CS" or dsName == "physics":
        dataset = Coauthor(path, name=dsName)
    elif dsName == "Actor":
        dataset = Actor(path)
    else:
        print("Dataset could not find, check spellings")
        exit()

    WithBatches(D, path, dsName, device, dataset, d)


if __name__ == '__main__':
    main()
