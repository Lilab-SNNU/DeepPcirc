import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score, \
    precision_recall_curve
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, StratifiedKFold
from DeepPcirc_model import *

def save_checkpoint(state,is_best,outfile):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, outfile)

    else:
        print("=> Validation Performance did not improve")


def ytest_ypred_to_file(y_test, y_pred, out_fn):
    with open(out_fn,'w') as f:
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred[i])+'\n')
def trainmodel(infile,outfile)
    data = pd.read_csv(infile, sep='\s+', header=None)
    print(data.shape, type(data))

    data_train = data.values.tolist()
    model_path = '.'

    X = np.array([_[1:] for _ in data_train])
    y = np.array([_[0] for _ in data_train])

    print(X.shape, type(X), y.shape, type(y))

    np.random.seed(12)
    np.random.shuffle(X)
    np.random.seed(12)
    np.random.shuffle(y)
    vec_len = 8

    best_acc_list = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0 / 8, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1] / vec_len), vec_len)
    X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1] / vec_len), vec_len)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    X_test = torch.from_numpy(X_test).float()
    X_train = X_train.reshape(X_train.shape[0], vec_len, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], vec_len, X_test.shape[1])
    print(X_train.shape, X_test.shape, type(X_train))

    wordvec_len = 64
    HIDDEN_NUM = 256
    LAYER_NUM = 2
    DROPOUT = 0.3
    cell = 'lstm'
    net = ronghe_model(wordvec_len, HIDDEN_NUM, LAYER_NUM, DROPOUT, cell)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = loss.to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)
    n_epochs = 80
    n_examples = len(X_train)
    BATCH_SIZE = 128
    patience = 0
    best_acc = 0
    for i in range(n_epochs):
        start_time = time.time()
        cost = 0.
        y_pred_prob_train = []
        y_batch_train = []
        y_batch_pred_train = []

        num_batches = n_examples // BATCH_SIZE
        for k in range(num_batches):
            start, end = k * BATCH_SIZE, (k + 1) * BATCH_SIZE
            output_train, y_pred_prob, y_batch, y_pred_train = train(net, loss, optimizer, X_train[start:end],
                                                                     y_train[start:end])

            cost += output_train

            prob_data = y_pred_prob.data.numpy()

            for m in range(len(prob_data)):
                y_pred_prob_train.append(np.exp(prob_data)[m][1])
            y_batch_train += y_batch
            y_batch_pred_train += y_pred_train

        scheduler.step()

        start, end = num_batches * BATCH_SIZE, n_examples
        output_train, y_pred_prob, y_batch, y_pred_train = train(net, loss, optimizer, X_train[start:end],
                                                                 y_train[start:end])

        cost += output_train

        prob_data = y_pred_prob.data.numpy()

        for m in range(len(prob_data)):
            y_pred_prob_train.append(np.exp(prob_data)[m][1])

        y_batch_train += y_batch
        y_batch_pred_train += y_pred_train
        fpr_train, tpr_train, thresholds_train = roc_curve(y_batch_train, y_pred_prob_train)

        output_test = predict(net, X_test)
        y_pred_prob_test = []
        y_pred_test = output_test.data.numpy().argmax(axis=1)

        prob_data = F.log_softmax(output_test, dim=1).data.numpy()
        for m in range(len(prob_data)):
            y_pred_prob_test.append(np.exp(prob_data)[m][1])

        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_prob_test)
        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)

        print(
            "Epoch %d, cost = %f, AUROC_train = %0.4f, train_acc = %.4f, train_recall= %.4f,train_precision = %.4f, train_f1score = %.4f,train_mcc= %.4f, test_acc = %.4f, test_recall= %.4f,test_precision = %.4f, test_f1score = %.4f,test_mcc= %.4f,AUROC_test = %0.4f"
            % (i + 1, cost / num_batches, auc(fpr_train, tpr_train), accuracy_score(y_batch_train, y_batch_pred_train),
               recall_score(y_batch_train, y_batch_pred_train), precision_score(y_batch_train, y_batch_pred_train),
               f1_score(y_batch_train, y_batch_pred_train),
               matthews_corrcoef(y_batch_train, y_batch_pred_train), accuracy_score(y_test, y_pred_test),
               recall_score(y_test, y_pred_test), precision_score(y_test, y_pred_test), f1_score(y_test, y_pred_test),
               matthews_corrcoef(y_test, y_pred_test),
               auc(fpr_test, tpr_test)))

        print("time cost: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        cur_acc = accuracy_score(y_batch_train, y_batch_pred_train)
        is_best = bool(cur_acc >= best_acc)
        best_acc = max(cur_acc, best_acc)

        save_checkpoint({
            'epoch': i + 1,
            'state_dict': net.state_dict(),
            'best_accuracy': best_acc,
            'optimizer': optimizer.state_dict()
        }, is_best, model_path)

        if not is_best:
            patience += 1
            if patience >= 10:
                break

        else:
            patience = 0

        if is_best:
            ytest_ypred_to_file(y_batch_train, y_pred_prob_train,
                                outfile)

            ytest_ypred_to_file(y_test, y_pred_prob_test,
                                outfile)


    best_acc_list.append(best_acc)
    print('> best acc:', best_acc)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-in", "--infile", action="store", dest='upfile', required=True,
                        help="coding file")
    parser.add_argument("-o", "--outfile", action="store", dest='outfile', required=True,
                        help="output model file")
    args = parser.parse_args()
    infile = args.infile
    outfile = args.outfile
    trainmodel(infile,outfile)
