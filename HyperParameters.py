import torch


class HyperParameters:

    def datasetName(self, dsName):
        if dsName == "Cora":
            Dict = self.CoraDataset(dsName)
        elif dsName == "CiteSeer":
            Dict = self.CiteSeerDataset(dsName)
        elif dsName == "PubMed":
            Dict = self.PubMedDataset(dsName)
        elif dsName == "dblp":
            Dict = self.DblpDataset(dsName)
        elif dsName == "WikiCS":
            Dict = self.WikiCSDataset(dsName)
        elif dsName == "photo":
            Dict = self.AmPhotoDataset(dsName)
        elif dsName == "computers":
            Dict = self.AmComputersDataset(dsName)
        elif dsName == "CS":
            Dict = self.COCSDataset(dsName)
        elif dsName == "Physics":
            Dict = self.COPhyDataset(dsName)
        elif dsName == "Actor":
            Dict = self.ActorDataset(dsName)
        else:
            print("Unknown Dataset")
        return Dict

    def CoraDataset(self, dsName):
        print("Executing Dataset:", dsName)
        coraDict = {
            "Base_Algo": "DAENS",
            "Encoder": "GAT",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 658942,
            "main_random_seed": 458642,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'Planetoid',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": 0,
            "drop_prob": 0,
            "percentage": 0.43,
            "activation": torch.nn.ReLU,
            "rounds": 10,
            "training_manual_seed": 761628,
            "training_random_seed": 951498,
            "Aug1_ER": 0.4,
            "Aug1_FM": 0.4,
            "Aug2_ER": 0.2,
            "Aug2_FM": 0.7,
            "hidden_dim": 128,
            "proj_dim": 128,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.4,
            "intraview_negs": False,
            "lr": 0.0005,
            "train_epochs": 1001,
            "test_epochs": 50,
            "Type": "1D2D",  # Both1D, 1D2D, Both2D

        }

        return coraDict

    def CiteSeerDataset(self, dsName):
        print("Executing Dataset:", dsName)
        Dict = {
            "Base_Algo": "DAENS",
            "Encoder": "GCN",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 5431411,
            "main_random_seed": 5545981,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'Planetoid',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": 0.4,
            "drop_prob": 0.3,
            "percentage": 0.375,
            "activation": torch.nn.PReLU,
            "rounds": 10,
            "training_manual_seed": 447643,
            "training_random_seed": 713487,
            "Aug1_ER": 0.4,
            "Aug1_FM": 0.4,
            "Aug2_ER": 0.2,
            "Aug2_FM": 0.8,
            "hidden_dim": 256,
            "proj_dim": 256,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.9,
            "intraview_negs": False,
            "lr": 0.00001,
            "train_epochs": 1101,
            "test_epochs": 50,
            "Type": "Both2D",  # Both1D, 1D2D, Both2D

        }

        return Dict

    def PubMedDataset(self, dsName):
        print("Executing Dataset:", dsName)
        Dict = {
            "Base_Algo": "DAENS",
            "Encoder": "GCN",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 65421,
            "main_random_seed": 499642,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'Planetoid',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": None,
            "drop_prob": 0.43,
            "percentage": 0.7,
            "activation": torch.nn.ReLU,
            "rounds": 10,
            "training_manual_seed": 512079,
            "training_random_seed": 40507,
            "Aug1_ER": 0.1,
            "Aug1_FM": 0.1,
            "Aug2_ER": 0.2,
            "Aug2_FM": 0.43,
            "hidden_dim": 256,
            "proj_dim": 128,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.4,
            "intraview_negs": False,
            "lr": 0.001,
            "train_epochs": 1551,
            "test_epochs": 50,
            "Type": "1D2D",  # Both1D, 1D2D, Both2D

        }

        return Dict

    def DblpDataset(self, dsName):
        print("Executing Dataset:", dsName)
        Dict = {
            "Base_Algo": "DAENS",
            "Encoder": "GCN",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 658942,
            "main_random_seed": 458642,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'CitationFull',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": None,
            "drop_prob": 0.3,
            "percentage": 0.8,
            "activation": torch.nn.ReLU,
            "rounds": 10,
            "training_manual_seed": 63742,
            "training_random_seed": 515793,
            "Aug1_ER": 0.1,
            "Aug1_FM": 0.1,
            "Aug2_ER": 0.2,
            "Aug2_FM": 0.375,
            "hidden_dim": 256,
            "proj_dim": 128,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.4,
            "intraview_negs": False,
            "lr": 0.001,
            "train_epochs": 1201,
            "test_epochs": 50,
            "Type": "Both2D",  # Both1D, 1D2D, Both2D

        }
        return Dict

    def WikiCSDataset(self, dsName):
        print("Executing Dataset:", dsName)
        Dict = {
            "Base_Algo": "DAENS",
            "Encoder": "GAT",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 496483,
            "main_random_seed": 347561,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'WikiCS',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": None,
            "drop_prob": 0.4,
            "percentage": 0.75,
            "activation": torch.nn.ReLU,
            "rounds": 10,
            "training_manual_seed": 494664,
            "training_random_seed": 507356,
            "Aug1_ER": 0.4,
            "Aug1_FM": 0.1,
            "Aug2_ER": 0.3,
            "Aug2_FM": 0.4,
            "hidden_dim": 256,
            "proj_dim": 128,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.4,
            "intraview_negs": False,
            "lr": 0.01,
            "train_epochs": 1801,
            "test_epochs": 50,
            "Type": "1D2D",  # Both1D, 1D2D, Both2D  # here results are same 1D2D and Both2D

        }
        return Dict

    def AmPhotoDataset(self, dsName):
        print("Executing Dataset:", dsName)
        Dict = {
            "Base_Algo": "DAENS",
            "Encoder": "GAT",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 465951,
            "main_random_seed": 436741,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'Amazon',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": None,
            "drop_prob": 0.8,
            "percentage": 0.375,
            "activation": torch.nn.ReLU,
            "rounds": 10,
            "training_manual_seed": 427352,
            "training_random_seed": 850847,
            "Aug1_ER": 0.5,
            "Aug1_FM": 0.1,
            "Aug2_ER": 0.3,
            "Aug2_FM": 0.8,
            "hidden_dim": 256,
            "proj_dim": 128,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.4,
            "intraview_negs": False,
            "lr": 0.01,
            "train_epochs": 1201,
            "test_epochs": 50,
            "Type": "1D2D",  # Both1D, 1D2D, Both2D

        }
        return Dict

    def AmComputersDataset(self, dsName):
        print("Executing Dataset:", dsName)
        Dict = {
            "Base_Algo": "DAENS",
            "Encoder": "GAT",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 465951,
            "main_random_seed": 436741,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'Amazon',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": None,
            "drop_prob": 0.7,
            "percentage": 0.43,
            "activation": torch.nn.ReLU,
            "rounds": 10,
            "training_manual_seed": 26612,
            "training_random_seed": 252361,
            "Aug1_ER": 0.5,
            "Aug1_FM": 0.1,
            "Aug2_ER": 0.5,
            "Aug2_FM": 0.7,
            "hidden_dim": 128,
            "proj_dim": 128,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.4,
            "intraview_negs": False,
            "lr": 0.001,
            "train_epochs": 1601,
            "test_epochs": 50,
            "Type": "Both2D",  # Both1D, 1D2D, Both2D

        }
        return Dict

    def COCSDataset(self, dsName):
        print("Executing Dataset:", dsName)
        Dict = {
            "Base_Algo": "DAENS",
            "Encoder": "GAT",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 102686,
            "main_random_seed": 445165,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'Coauthor',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": None,
            "drop_prob": 0.75,
            "percentage": 0.4,
            "activation": torch.nn.ReLU,
            "rounds": 10,
            "training_manual_seed": 865777,
            "training_random_seed": 263941,
            "Aug1_ER": 0.4,
            "Aug1_FM": 0.3,
            "Aug2_ER": 0.2,
            "Aug2_FM": 0.75,
            "hidden_dim": 256,
            "proj_dim": 128,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.4,
            "intraview_negs": False,
            "lr": 0.0005,
            "train_epochs": 1701,
            "test_epochs": 50,
            "Type": "1D2D",  # Both1D, 1D2D, Both2D

        }
        return Dict

    def COPhyDataset(self, dsName):
        print("Executing Dataset:", dsName)
        Dict = {
            "Base_Algo": "DAENS",
            "Encoder": "GAT",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 496443,
            "main_random_seed": 34461,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'Coauthor',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": None,
            "drop_prob": 0.3,
            "percentage": 1,
            "activation": torch.nn.ReLU,
            "rounds": 10,
            "training_manual_seed": 494664,
            "training_random_seed": 507356,
            "Aug1_ER": 0.4,
            "Aug1_FM": 0.1,
            "Aug2_ER": 0.3,
            "Aug2_FM": 0.3,
            "hidden_dim": 256,
            "proj_dim": 128,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.4,
            "intraview_negs": False,
            "lr": 0.001,
            "train_epochs": 1801,
            "test_epochs": 50,
            "Type": "1D2D",  # Both1D, 1D2D, Both2D
        }

        return Dict

    def ActorDataset(self, dsName):
        print("Executing Dataset:", dsName)
        Dict = {
            "Base_Algo": "DAENS",
            "Encoder": "GAT",
            "Dataset": dsName,
            "TSNE": "NO",
            "main_manual_seed": 6672,
            "main_random_seed": 4508,
            "dataset_P_folder": 'datasets',
            "dataset_C_folder": 'Actor',
            "batches": 'no',
            "cluster_num_parts": 10,
            "cluster_batch_size": 5,
            "drop_prob_1": None,
            "drop_prob": 0,
            "percentage": 0.6,
            "activation": torch.nn.ReLU,
            "rounds": 10,
            "training_manual_seed": 6083,
            "training_random_seed": 1238,
            "Aug1_ER": 0.4,
            "Aug1_FM": 0.4,
            "Aug2_ER": 0.2,
            "Aug2_FM": 0.5,
            "hidden_dim": 128,
            "proj_dim": 128,
            "input_dim": 0,
            "num_layers": 2,
            "tau": 0.2,
            "intraview_negs": False,
            "lr": 0.0005,
            "train_epochs": 1401,
            "test_epochs": 50,
            "Type": "Both2D",  # Both1D, 1D2D, Both2D

        }
        return Dict