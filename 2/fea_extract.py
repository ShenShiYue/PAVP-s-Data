#导入库
from Bio import SeqIO
import pandas as pd
import re, math
from itertools import *
SPLITSEED = 810
from sklearn.decomposition import KernelPCA as KPCA

_AALetter = ['A', 'C', 'D', 'E', 'F', 'G', 'H',
             'I', 'K', 'L', 'M', 'N', 'P', 'Q',
             'R', 'S', 'T', 'V', 'W', 'Y']
_ftypes = ["AAC", "DiAAC", "PHYCs"]

word2int1DDict = {'G': 1, 'A': 2, 'V': 3, 'L': 4, 'I': 5, 'P': 6,
                               'F': 7, 'Y': 8, 'W': 9, 'S': 10, 'T': 11, 'C': 12,
                               'M': 13, 'N': 14, 'Q': 15, 'D': 16, 'E': 17,
                               'K': 18, 'R': 19, 'H': 20, 'X': 21, 'B': 22,
                               'J': 23, 'O': 24, 'U': 25, 'Z': 26}

int1D2wordDict = {1: 'G',2: 'A',3: 'V',4: 'L', 5: 'I',6: 'P',7: 'F', 8: 'Y',
                  9: 'W', 10: 'S', 11: 'T', 12: 'C', 13: 'M', 14: 'N', 15: 'Q',
                  16: 'D', 17: 'E', 18: 'K', 19: 'R', 20: 'H', 21: 'X', 22: 'B',
                  23: 'J', 24: 'O', 25: 'U', 26: 'Z'}

#读取文件
def read_fasta(fname):
    with open(fname, "r") as f:
        seq_dict = [(record.id, record.seq._data) for record in SeqIO.parse(f, "fasta")]
    seq_df = pd.DataFrame(data=seq_dict, columns=["Id", "Sequence"])
    return seq_df

#AAC
def insert_AAC(seq_df):
    # Compute AAC for peptide in specific A.A
    def get_aac(seq, aa):
        return seq.count(aa) / len(seq) * 100

    # processing data_frame
    data_size = seq_df.size
    for ll in _AALetter:
        seq_df['AAC_{}'.format(ll)] = list(map(get_aac, seq_df['Sequence'], [ll] * data_size))
    return seq_df

#Kmer_KPCA
#定义Kmer函数
def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group)  # Alphabetically sorted list
        for c in g_members:
            # print('c' + str(c))
            # print('g_members[0]' + str(g_members[0]))
            result[c] = str(tar_list[index])  # K:V map, use group's first letter as represent.
        index = index + 1
    return result

def translate_sequence(seq, TranslationDict):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    import string
    from_list = []
    to_list = []
    for k, v in TranslationDict.items():#遍历TD
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    # TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]#求模运算，相当于mod，也就是计算除法的余数
        n = n / base
        ch1 = chars[int(n % base)]
        n = n / base
        ch2 = chars[int(n % base)]
        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com

def get_4_nucleotide_composition_KPCA(tris, seq, pythoncount=True):
    seq_len = len(seq)
    tri_feature = [0] * len(tris)
    k = len(tris[0])
    note_feature = [[0 for cols in range(len(seq) - k + 1)] for rows in range(len(tris))]
    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num) / seq_len)
    else:
        # tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1 - k):
            kmer = seq[x:x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                # tmp_fea[ind] = tmp_fea[ind] + 1
                note_feature[ind][x] = note_feature[ind][x] + 1
        estimator = KPCA(n_components=1, kernel='rbf', gamma=15)
        tri_feature = estimator.fit_transform(note_feature).T
        tri_feature = tri_feature.flatten()
    # print tri_feature
        # pdb.set_trace()
    return tri_feature

def Kmer_KPCA(seq_df):
    encoding = []
    heard = []
    for i in range(343):
        heard.append("Kmer_KPCA_" + str(i))
    encoding.append(heard)
    protein_tris = get_3_protein_trids()
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    for seq in seq_df:
        protein_seq = translate_sequence(seq, group_dict)
        protein_tri_fea = get_4_nucleotide_composition_KPCA(protein_tris, protein_seq, pythoncount =False)
        protein_tri_fea = list(protein_tri_fea)
        encoding.append(protein_tri_fea)
    return encoding

def insert_Kmer_KPCA(seq_df):
    encoding = Kmer_KPCA(seq_df["Sequence"])
    encoding = pd.DataFrame(encoding[1:], columns=encoding[0])
    seq_df = pd.concat([seq_df, encoding],axis=1)
    return seq_df


#  CKSAAGP

def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair

def minSequenceLength(fastas):#查看最小长度
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen

def cksaagp(fastas, gap = 3, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fastas) < gap+2:
        print('Error: all the sequence length should be greater than the (gap value) + 2 = ' + str(gap+2) + '\n\n')
        return 0

    group = {'alphaticr': 'GAVLMI',
             'aromatic': 'FYW',
             'postivecharger': 'KRH',
             'negativecharger': 'DE',
             'uncharger': 'STCPNQ'}

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1+'.'+key2)

    encodings = []
    header = ['#']
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p+'.gap'+str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] = gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)

        encodings.append(code)

    return encodings

def insert_CKSAAGP(seq_df, gap=2):
    fastas = [[idx, seq] for idx, seq in zip(seq_df['Id'], seq_df['Sequence'])]
    encoding = cksaagp(fastas, gap=gap)
    encoding = pd.DataFrame(encoding[1:], columns=encoding[0])
    seq_df = pd.concat([seq_df, encoding.iloc[:, 1:]], axis=1)
    return seq_df

#CTD

# CTD_function
group1 = {'hydrophobicity_PRAM900101': 'RKEDQN', 'normwaalsvolume': 'GASTPDC', 'polarity': 'LIFWCMVY',
          'polarizability': 'GASDT', 'charge': 'KR', 'secondarystruct': 'EALMQKRH', 'solventaccess': 'ALFCGIVW'}
group2 = {'hydrophobicity_PRAM900101': 'GASTPHY', 'normwaalsvolume': 'NVEQIL', 'polarity': 'PATGS',
          'polarizability': 'CPNVEQIL', 'charge': 'ANCQGHILMFPSTWYV', 'secondarystruct': 'VIYCWFT',
          'solventaccess': 'RKQEND'}
group3 = {'hydrophobicity_PRAM900101': 'CLVIMFW', 'normwaalsvolume': 'MHKFRYW', 'polarity': 'HQRKNED',
          'polarizability': 'KMHFRYW', 'charge': 'DE', 'secondarystruct': 'GNPSD', 'solventaccess': 'MSPTHY'}
groups = [group1, group2, group3]
propertys = ('hydrophobicity_PRAM900101', 'normwaalsvolume', 'polarity', 'polarizability', 'charge', 'secondarystruct',
             'solventaccess')

def Count_C(sequence1, sequence2):
    sum = 0
    for aa in sequence1:
        sum = sum + sequence2.count(aa)
    return sum

def Count_D(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]
    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence))
                    break
        if myCount == 0:
            code.append(0)
    return code
def CTD(seqs):
    encodings = []
    header = []
    for i in range(147):
        header.append('CTD_' + str(i))
    encodings.append(header)
    for seq in seqs:
        code = []
        code2 = []
        CTDD1 = []
        CTDD2 = []
        CTDD3 = []
        aaPair = [seq[j:j + 2] for j in range(len(seq) - 1)]
        for p in propertys:
            c1 = Count_C(group1[p], seq) / len(seq)
            c2 = Count_C(group2[p], seq) / len(seq)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]

            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code2 = code2 + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
            CTDD1 = CTDD1 + [value / float(len(seq)) for value in Count_D(group1[p], seq)]
            CTDD2 = CTDD2 + [value / float(len(seq)) for value in Count_D(group2[p], seq)]
            CTDD3 = CTDD3 + [value / float(len(seq)) for value in Count_D(group3[p], seq)]
        encodings.append(code + code2 + CTDD1 + CTDD2 + CTDD3)
    return encodings

def insert_CTD(seq_df):
    enconding = CTD(seq_df['Sequence'])
    enconding = pd.DataFrame(enconding[1:], columns = enconding[0])
    seq_df = pd.concat([seq_df, enconding.iloc[:, :]],axis = 1)
    return seq_df

#AAE
def AAE_1(fastas):
    length = float(len(fastas))
    amino_acids = dict.fromkeys(_AALetter, 0)
    encodings = []
    for AA in amino_acids:
        hits = [a.start() for a in list(re.finditer(AA, fastas))]
        p_prev = 0
        p_next = 1
        sum = 0
        while p_next < len(hits):
            distance = (hits[p_next] - hits[p_prev]) / length
            sum += distance * math.log(distance, 2)
            p_prev = p_next
            p_next += 1
        amino_acids[AA] = -sum
        encodings.append(amino_acids[AA])
    return encodings

def AAE(seq):
    encodings = []
    header = []
    for i in range(60):
        header.append('AAE_' + str(i))
    encodings.append(header)
    for fastas in seq:
        fastas_NT5 = "%s" % fastas[:5]
        fastas_CT5 = "%s" % fastas[-5:]
        encodings_full = AAE_1(fastas)
        encodings_CT5 = AAE_1(fastas_CT5)
        encodings_NT5 = AAE_1(fastas_NT5)
        encodings.append(encodings_full + encodings_NT5 + encodings_CT5)
    return encodings

def insert_AAE(seq_df):
    encoding = AAE(seq_df['Sequence'])
    encoding = pd.DataFrame(encoding[1:], columns=encoding[0])
    seq_df = pd.concat([seq_df, encoding.iloc[:, :]], axis=1)
    return seq_df

#ASDC
"""ASDC"""
Amino_acids = ['A','C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
'R', 'S','T','V', 'W','Y']
Amino_acids_ = list(product(Amino_acids,Amino_acids))
Amino_acids_ = [i[0]+i[1] for i in Amino_acids_]

def ASDC(seqs):
    header = []
    for i in range(400):
        header.append('ASDC_'+ str(i))
    seqs_ = []
    seqs_.append(header)
    for seq in seqs:
        ASDC_feature = []
        skip = 0
        for i in range(len(seq)): 
            ASDC_feature.extend(Skip(seq,skip)) 
            skip+=1
        seqs_.append([ASDC_feature.count(i)/len(ASDC_feature) for i in Amino_acids_])
    return seqs_

def Skip(seq,skip):
	element = []
	for i in range(len(seq)-skip-1):
		element.append(seq[i]+seq[i+skip+1])
	return element

def insert_ASDC(seq_df):
    encoding = ASDC(seq_df['Sequence'])
    encoding = pd.DataFrame(encoding[1:], columns=encoding[0])
    seq_df = pd.concat([seq_df, encoding.iloc[:, :]], axis=1)
    return seq_df