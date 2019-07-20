# template to normalise raw counts

library(DESeq2)
mx_samples = read.table('../test_data/GSE80655_GeneExpressionData_Updated_3-26-2018.txt', header=TRUE, sep = '\t')

# convert to numeric
countDataMatrix <- as.matrix(mx_samples[ , -1])
rownames(countDataMatrix) <- mx_samples[ , 1]

# construct fake design
sample_names = colnames(countDataMatrix)
mx_design = as.matrix(rbinom(length(sample_names), 1, 0.5))
colnames(mx_design) <- 'disease'
rownames(mx_design) <- sample_names


# create DESeq2 structure
dds <- DESeqDataSetFromMatrix(countData  = countDataMatrix,
                              colData = mx_design,
                              design =~ disease)

keep <- rowSums(counts(dds)) >= 20
dds <- dds[keep,]

# estimate normalising factors
dds <- estimateSizeFactors(dds)

# get normalised matrix of samples
mx_samples_normalised = counts(dds, normalized=TRUE)

# convert from Ensembl to Gene Symbol
library(biomaRt)

mart = useEnsembl(biomart="ensembl", dataset="hsapiens_gene_ensembl")
new_names <- getBM(filters = "ensembl_gene_id", 
                   attributes = c("ensembl_gene_id", "hgnc_symbol"),
                   values=rownames(mx_samples_normalised), 
                   mart= mart)

#here comes comlete mess
indices <- which(new_names$hgnc_symbol != "")
filtered <- mx_samples_normalised[indices,,drop=F]
names = new_names[indices,]
names <- names[ , 2]
row.names(filtered) = names

# SAVE
write.table(t(filtered), file = '../test_data/80655_norm.txt', quote = FALSE, sep = ' ', eol = '\n')

