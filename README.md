# DeepPcirc
Predict Plant Circular RNA Based on Deep Learning

---

## Requirement
### Data:

- Genome sequence (fasta format)
- Genome annotation file (gtf/gff format)
- RNA_seq data (fastq format)
### Software:

- NCBI-blast(v.2.9.0+):(ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
- bowtie2 (v.2.2.6+): (https://github.com/BenLangmead/bowtie2)
- tophat2 (v.2.1.1+): (http://ccb.jhu.edu/software/tophat)
- samtools (v.0.1.19+): (https://github.com/samtools/samtools)
- Python3 (v.3.7.7): (https://www.python.org/)
- iLrearn: (https://github.com/Superzchen/iLearn)
### Python3 package:

- Biopython (v.1.76): (https://pypi.org/project/biopython/)
- Scikit-learn (v.0.23.2): (https://scikit-learn.org/stable/)
- Pytorch (v.1.6.0): (https://pytorch.org/)

---
## Download
  Open the terminal and input:
  ```bash
  git clone https://github.com/Lilab-SNNU/DeepPcirc.git
  ```
---
## Usage

### - You can run DeepPcirc step by step use command line

1. Alignment

     In this step need given file:

     - Genome sequence (fasta format)
     - Genome annotation file (gtf/gff format)
     - RNA_seq data (fastq format)

     Then, you will get a file named ***unmapped.fasta*** in ***output_dir***, it content the information of unmapped reads and will be used to the ***step Blast*** 
  ```bash
   python DeepPcirc_aligment.py -g genome.fa -G genome.gtf -o output_dir <reads_1[ reads_2]>
  ```

  2. Blast

     Blast alignment of unmapped reads sequence and genomic sequences, in this step need given file:

     - Genome sequence (fasta format)
     - Unmapped reads from ***step Alignment*** (fasta format)

     Then, you will get a file named unmapped.blast in the dir which unmapped.fasta in, it content the blast information and will be used to the ***step Sequence extraction***
  ```bash
   python DeepPcirc_unmapped_reads_blast.py  -g genome.fa -q unmapped.fasta -o output_dir
  ```
   3.Sequence extraction
      Extracting feature sequences,in this step need given file:
      - blast information file from ***step Blast***
      - Genome sequence (fasta format) 
      Then you can get ***upfile*** and **downfile***and will be used to the ***step Reads coding***

  ```bash
   python DeepPcirc_getseq.py -i unmapped.blast -g genome.fa
  ```
   4.Reads coding
      Encode the sequence,in this step need given file:
      - up reads file
      - down reads file 
      Then you can get a file named ***pre_coding*** and will be used to the ***step Circular RNA predicting***

  ```bash
   python DeepPcirc_seqprocess.py -up upfile -down downfile -out out
  ```
   5.Circular RNA predicting
      Final this step, you will get the circular RNA information saved in a file name ***pre_result***. 
      
  ```bash
   python DeepPcirc_predict.py -infile pre_coding  -modelfile osamodel.pt -outfile pre_result
  ```
   6.train new model
      If you have new species data, you can train new models yourself. 
      
  ```bash
   python DeepPcirc_train.py -infile train_coding  -o model
  ```
---
## Contact us

If you encounter any problems while using Pcirc, please send an email (glli@snnu.edu.cn) or submit the issues on GitHub (https://github.com/Lilab-SNNU/DeepPcirc/issues) and we will resolve it as soon as possible.
