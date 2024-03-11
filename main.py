"""Running GWNN."""

from gwnn import GWNNTrainer
from utils import WaveletSparsifier
from param_parser import parameter_parser
from utils import tab_printer, graph_reader, feature_reader, target_reader, save_logs
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold


def main():
    """
    Parsing command line parameters, reading data.
    Doing sparsification, fitting a GWNN and saving the logs.
    """
    args = parameter_parser()
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    features = np.genfromtxt('./input_ASD/features.txt')
    features_coo = sparse.coo_matrix(features)
    target = target_reader(args.target_path)
    sparsifier = WaveletSparsifier(graph, args.scale, args.approximation_order, args.tolerance)
    sparsifier.calculate_all_wavelets()
    skf = StratifiedKFold(n_splits=args.folds)
    cv_splits = list(skf.split(features, target))
    score ={}
    score['acc'] = 0
    score['auc'] = 0
    score['pre'] = 0
    score['recall'] = 0
    score['F1'] = 0
    for i in range(args.folds):
        print(f"--------------Folds:{i} Start---------")
        trainer = GWNNTrainer(args, sparsifier, features_coo, target,train_nodes=cv_splits[i][0], test_nodes=cv_splits[i][1])
        trainer.fit()
        a = trainer.score()
        score['acc'] += a['acc']
        score['auc'] += a['auc']
        score['pre'] += a['pre']
        score['recall'] += a['recall']
        score['F1'] += a['F1']


    # trainer = GWNNTrainer(args, sparsifier, features_coo, target)
    # trainer.fit()
    # trainer.score()
    save_logs(args, trainer.logs)

    print('---------Mean Results------------')
    print('Acc:{:.4f}'.format(score['acc']/args.folds),
        'Pre:{:.4f}'.format(score['pre']/args.folds),
        'Recall:{:.4f}'.format(score['recall']/args.folds),
        'F1:{:.4f}'.format(score['F1']/args.folds),
        'AUC:{:.4f}'.format(score['auc']/args.folds),

          )
    
if __name__ == "__main__":
    main()
