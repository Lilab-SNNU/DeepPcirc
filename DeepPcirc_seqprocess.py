from Bio import SeqIO
import pandas as pd
import os
import argparse


def coding(upfile, downfile):
    ilearn_path = os.popen('locate iLearn-nucleotide-basic.py').read().split()[0]
    out_path = '/'.join(upfile.split('/')[:-1])
    upNCP = out_path + '/upNCP'
    upANF = out_path + '/upANF'
    downNCP = out_path + '/downNCP'
    downANF = out_path + '/downANF'
    ilearn_upNCP = os.system("python '%s' --file %s --method NCP --format tsv --out %s" % (ilearn_path, upfile, upNCP))
    ilearn_upANF = os.system("python '%s' --file %s --method ANF --format tsv --out %s" % (ilearn_path, upfile, upANF))
    ilearn_downNCP = os.system(
        "python '%s' --file %s --method NCP --format tsv --out %s" % (ilearn_path, downfile, downNCP))
    ilearn_downANF = os.system(
        "python '%s' --file %s --method ANF --format tsv --out %s" % (ilearn_path, downfile, downANF))

    return upNCP, upANF, downNCP, downANF


def codingadd(upfile, downfile): 
    upNCP, upANF, downNCP, downANF = coding(upfile, downfile)
    upNCP_pd = pd.read_csv(upNCP, sep='\s+', header=None)
    upANF_pd = pd.read_csv(upANF, sep='\s+', header=None)
    downNCP_pd = pd.read_csv(downNCP, sep='\s+', header=None)
    downANF_pd = pd.read_csv(downANF, sep='\s+', header=None)
    up_NCP_ANF = upNCP_pd.iloc[:, 0]
    upANF_pd = upANF_pd.iloc[:,1:]
    downANF_pd = downANF_pd.iloc[:,1:]
    down_NCP_ANF = downNCP_pd.iloc[:, 0]
    out_path = '/'.join(upfile.split('/')[:-1])
    up_path = out_path + '/up_NCP_ANF_out'
    down_path = out_path + '/down_NCP_ANF_out'
    up_NCP_ANF_out = open(up_path, 'w')
    down_NCP_ANF_out = open(down_path, 'w')
    down_NCP_ANF_out_nolabel = out_path + '/up_NCP_ANF_out_nolabel'
    pre_NCP_ANF = out_path + '/pre_coding'
    for a in range(0, len(upANF_pd.iloc[1, :])):
        up_NCP_ANF = pd.concat([up_NCP_ANF,
                                upNCP_pd.iloc[:, 1 + (3 * a)],
                                upNCP_pd.iloc[:, 2 + (3 * a)],
                                upNCP_pd.iloc[:, 3 + (3 * a)],
                                upANF_pd.iloc[:, a]],
                               axis=1)
    for a in range(0, len(downANF_pd.iloc[1, :])):
        down_NCP_ANF = pd.concat([down_NCP_ANF,
                                  downNCP_pd.iloc[:, 1 + (3 * a)],
                                  downNCP_pd.iloc[:, 2 + (3 * a)],
                                  downNCP_pd.iloc[:, 3 + (3 * a)],
                                  downANF_pd.iloc[:, a]],
                                 axis=1)
    up_NCP_ANF.to_csv(up_NCP_ANF_out, sep='\t', header=None, index=None)
    down_NCP_ANF.to_csv(down_NCP_ANF_out, sep='\t', header=None, index=None)
    up_NCP_ANF_out.close()
    down_NCP_ANF_out.close()
    os.system("awk '{$1=\"\";print $0}' %s > %s" %(down_path,down_NCP_ANF_out_nolabel))
    os.system("paste %s %s > %s" %(up_path,down_NCP_ANF_out_nolabel,pre_NCP_ANF))
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-up", "--upfile", action="store", dest='upfile', required=True,
                        help="up reads")
    parser.add_argument("-down", "--downfile", action="store", dest='downfile', required=True,
                        help="down reads")
    args = parser.parse_args()
    upfile = args.upfile
    downfile = args.downfile
    codingadd(upfile, downfile)

