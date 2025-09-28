import numpy as np
from numpy import ma
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from textwrap import dedent
import pandas as pd
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# from RAG import flat_docs


# from RAG import splitter


# def chunk_text(text, chunk_size=500, chunk_overlap=100):
    # words = str(text).split()
    # chunks = []
    #
    # for i in range(0, len(words), chunk_size - chunk_overlap):
    #     chunk = ' '.join(words[i: i + chunk_size])
    #     chunks.append(chunk)
    #
    # return chunks


class HousingBankRAG:

    def __init__(self,model_id = "LiquidAI/LFM2-2.6B"):


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()

        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2',device='cpu')

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=['\n\n', '\n', ' ', '']
        )

        df = pd.read_csv('full_housing_eda.csv')
        texts = df['Text'].to_list()
        texts = [str(text) for text in texts]
        big_text = '\n\n'.join(texts)

        self.docs = self.splitter.split_text(big_text)
        # docs = list(map(chunk_text, texts))

        tokenized_docs = [d.split() for d in self.docs]
        self.bm25 = BM25Okapi(tokenized_docs)

        # flat_docs = [chunk for doc in self.docs for chunk in doc]
        flat_docs = self.docs
        doc_embeddings = self.embed_model.encode(self.docs, convert_to_numpy=True)

        faiss.normalize_L2(doc_embeddings)

        self.index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        self.index.add(doc_embeddings)
        self.id2doc = {i: flat_docs[i] for i in range(len(flat_docs))}

        torch.cuda.empty_cache()

        self.model_id = model_id
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     device_map=None,
        #     dtype="bfloat16",
        #     #    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = None


    def retrieve(self, query, top_k=3,alpha = 0.5):
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        D, I = self.index.search(q_emb, top_k)
        faiss_scores = np.zeros(len(self.docs))

        for idx, score in zip(I[0],D[0]):
            faiss_scores[idx] = 1 + (score + 1e-9)

        bm25_scores = self.bm25.get_scores(query.split())

        faiss_scores = (faiss_scores - faiss_scores.min()) / (faiss_scores.max() - faiss_scores.min() + 1e-9)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)


        hybrid_scores = alpha * faiss_scores + (1 - alpha) * bm25_scores

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [self.docs[i] for i in top_indices]



    def generate(self,prompt,max_new_tokens = 512):

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map='cuda',
                dtype="bfloat16",
                #    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
            ).to('cuda')

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            # tokenize=True,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(input_ids,
                                         do_sample=True,
                                         temperature=0.3,
                                         min_p=0.15,
                                         repetition_penalty=1.05,
                                        max_new_tokens=max_new_tokens,
                                         )

        raw_output = self.tokenizer.decode(output[0], skip_special_tokens=False)
        # print(raw_output)
        matches = re.findall(r"<\|im_start\|>assistant\s*(.*?)(?=<\|im_end\|>)", raw_output, re.S)

        # self.model.to('cpu')
        # torch.cuda.empty_cache()

        if matches:
            return matches[-1].strip()
        return None


    def augment(self,data_row):
        prompt = dedent(f"""
        You are a helpful assistant.
        Only use the information provided below to answer the question,
        If the information does not contain the answer, reply strictly with "I don't Know :-( ", no more, no less, and don't add anything else to the output
        Question: {data_row['question']}
    
        Information:
    
        ```
        {data_row['context']}
        ```
        """)
        messages = [
            {"role": "system", "content": "Use only the information to answer the question"},
            {"role": "user", "content": prompt},
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    def search_question(self,query,top_k=3,max_new_tokens=1024,alpha = 0.5):

        # results = self.retrieve(query)
        data_row = {
            'question': query,
            'context': '\n'.join(self.retrieve(query,top_k,alpha))
        }

        prompt = self.augment(data_row)

        return self.generate(prompt,max_new_tokens)


