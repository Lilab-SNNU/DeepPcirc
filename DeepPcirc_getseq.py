import argparse
import multiprocessing
import os
import pickle
import pandas as pd
import Pcirc_deal_blast_result
import Pcirc_circ_seq_extract


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

