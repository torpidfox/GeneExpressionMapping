# template to normalise raw counts

library(DESeq2)
mx_samples = read.table('C:/r/GSE89403_rawCounts_GeneNames_AllSamples.txt', header=TRUE, sep = ',')

# construct fake design
sample_names = colnames(mx_samples)
mx_design = as.matrix(rbinom(length(sample_names), 1, 0.5))
colnames(mx_design) <- 'disease'
rownames(mx_design) <- sample_names

# create DESeq2 structure
dds <- DESeqDataSetFromMatrix(countData  = mx_samples,
                              colData = mx_design,
                              design =~ disease)
# estimate normalising factors
dds <- estimateSizeFactors(dds)

# get normalised matrix of samples
mx_samples_normalised = counts(dds, normalized=TRUE)

# convert from Ensembl to Gene Symbol
library(biomaRt)

mart = useEnsembl(biomart="ensembl", dataset="hsapiens_gene_ensembl")
new_names <- getBM(filters = "ensembl_gene_id", attributes = c("ensembl_gene_id", "hgnc_symbol"), values=rownames(mx_samples_normalised), mart= mart)

#here comes comlete mess
indices <- which(new_names$hgnc_symbol != "")
key1 <- mx_samples_normalised[indices,,drop=F]
names = new_names[indices, drop = T]
names <- names[ , 2]
row.names(key1) = names

# SAVE
write.table(mx_samples_normalised, file = 'preprocessed_data_89403.txt', quote = FALSE, sep = ' ', eol = '\n')

