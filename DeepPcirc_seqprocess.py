from Bio import SeqIO
import pandas as pd
import os
import argparse
import DeepPcirc_deal_blast_result
import DeepPcirc_seq_extract
import multiprocessing

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


    path = os.path.realpath(__file__)  
    abs_dir = path[:path.rfind('/')]
    parser = argparse.ArgumentParser()

    parser.add_argument("-i",
                        help='path to the result of blast, and the file format please refer to the format of '
                             'blast output format 6, you can refer to the test.blast')
    parser.add_argument("-g",
                        help='path to genome, and the file format is fasta')


    args = parser.parse_args() 
    info_file = args.i  
    genome_file = args.g  


    alignment_res = '/'.join(info_file.split('/')[:-1]) + '/alignment_res'
    df_alis = Pcirc_deal_blast_result.exactReads(info_file)
    df_list = Pcirc_deal_blast_result.cutDatafram(df_alis)
    pool = multiprocessing.Pool(len(df_list))
    print('Alignment reads begining')

    for dataframe in df_list:
        pool.apply_async(Pcirc_deal_blast_result.getCircInformation,
                         (dataframe, alignment_res, ))
    pool.close()
    pool.join()
    print('Alignment reads ending')

    up_filename, down_filename = Pcirc_circ_seq_extract.get_seq(alignment_res, genome_file)
    codingadd(up_filename, down_filename)

