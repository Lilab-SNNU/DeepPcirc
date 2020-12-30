# DeepPcirc
Predict Plant Circular RNA Based on Deep Learning

---

## Requirement
### Data:

- Genome sequence (fasta format)
- Genome annotation file (gtf/gff format)
- RNA_seq data (fastq format)
### Software:

#### Alignment

- NCBI-blast(v.2.9.0+):(ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
- bowtie2 (v.2.2.6+): (https://github.com/BenLangmead/bowtie2)
- tophat2 (v.2.1.1+): (http://ccb.jhu.edu/software/tophat)
- samtools (v.0.1.19+): (https://github.com/samtools/samtools)
- Python3 (v.3.7.7): (https://www.python.org/)
- iLrearn: (https://github.com/Superzchen/iLearn)
### Python3 package:

- Biopython (v.1.76): (https://pypi.org/project/biopython/)
- Scikit-learn (v.0.23.2): (https://scikit-learn.org/stable/)
- Pytorch(1.6.0): (https://pytorch.org/)

---
## Download
  Open the terminal and input:
  ```bash
  git clone https://github.com/Lilab-SNNU/DeepPcirc.git
  ```
---
## Usage

### - You can run DeepPcirc step by step use command line

    1.
    
    2.Reads coding
      You got ***upreads.fa*** and ***downreads.fa*** from the previous step.In order to make predictions, you need to code them.In this step, you enter the two files obtained in the previous step,then you can get a file named ***pre_coding***.
  ```bash
   python DeepPcirc_seqprocess.py -up upreads.fa -down downreads.fa
  ```
    3.Circular RNA predicting
      Final this step, you will get the circular RNA information saved in a file name circ. 
      
  ```bash
   python DeepPcirc_predict.py -infile pre_coding  -modelfile checkpointNCP_ANF.pt -outfile pre_result
  ```
