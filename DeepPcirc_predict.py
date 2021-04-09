from DeepPcirc_model import *
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-infile", "--infile", action="store", dest='infile', required=True,
                        help="predict coding file")
    parser.add_argument("-modelfile", "--model", action="store", dest='model', required=True,
                        help="model")
    parser.add_argument("-outfile", "--outfile", action="store", dest='outfile', required=True,
                        help="outfile name")

    args = parser.parse_args()
    infile = args.infile
    modelpt = args.model
    checkpoint = torch.load(modelpt)
    wordvec_len = 64
    HIDDEN_NUM = 256
    LAYER_NUM = 2
    DROPOUT = 0.3
    cell = 'lstm'
    vec_len = 8
    model = ronghe_model(wordvec_len, HIDDEN_NUM, LAYER_NUM, DROPOUT, cell)
    model.load_state_dict(checkpoint['state_dict'])

    data = pd.read_csv(infile,sep='\s+',header=None)
    data_pred = data.values.tolist()
    X = np.array([_[:] for _ in data_pred])
    X = X.reshape(X.shape[0],int(X.shape[1]/vec_len),vec_len)
    X = torch.from_numpy(X).float()
    X = X.reshape(X.shape[0],vec_len,X.shape[1])
    print(X.shape)

    i = 0
    N = X.shape[0]
    BATCH_SIZE = 128
    N =len(X)
    with open(args.outfile,'w') as fw:
        while i +  BATCH_SIZE < N:
            x_batch = X[i:i+BATCH_SIZE]
        
            output_test = predict(model,x_batch)

            prob_data = F.log_softmax(output_test, dim=1).data.numpy()
            for m in range(len(prob_data)):
                fw.write(str(np.exp(prob_data)[m][1])+'\n')
            i += BATCH_SIZE
        x_batch = X[i:N]
        output_test = predict(model,x_batch)
        prob_data = F.log_softmax(output_test, dim=1).data.numpy()
        for m in range(len(prob_data)):
            fw.write(str(np.exp(prob_data)[m][1])+'\n')

