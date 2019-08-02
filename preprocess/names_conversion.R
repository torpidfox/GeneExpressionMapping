genes = scan('../test_data/gene_names.txt', character())
genes

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("hgu133a.db")

library(hgu133a.db)
library(annotate)
x <- hgu133aSYMBOL
mapped_probes <- mappedkeys(x)
genesym.probeid <- as.data.frame(x[mapped_probes])
head(genesym.probeid)

library(data.table)
mapped = setDT(genesym.probeid, key = 'probe_id')[J(genes)]
write.table(mapped, file='../test_data/affy_to_kegg_2', quote=FALSE, row.names=FALSE)
