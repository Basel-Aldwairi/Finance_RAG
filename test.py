from torch import export

import RAG
# export.export()CUDA_VISIBLE_DEVICES=0

rag = RAG.HousingBankRAG()

print(rag.search_question('How can i get a loan?'))