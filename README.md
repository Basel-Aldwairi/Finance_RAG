# Finance_RAG
### Basel Al-Dwairi

___

### Retrieval-Augmented Generation for Financial Institutes (Housing Bank of Jordan)

### Capstone Project for AI Engineering Intern
___


### Features:

- Custom Data Collection Script:
  - Focused Crawler: 
    - Gets all URLs from base URL in BFT
    - Saves Names of Pages, and URLs in .csv
  - Web Scraper: 
    - Scraps HTML
    - Saves Names of Pages, URls, and HTML in .csv
  - EDA:
    - Cleans Data with custom functions
    - Saves Names of pages, URLs, and Clean Data in .csv
- RAG:
  - LLM:
    - LiquidAI/LFM2-2.6B: 2.6 Billion Parameters
  - Embedding:
    - SentenceTransformer: all-MiniLM-L6-v2
  - Chunking:
    - RecursiveCharacterTextSplitter
  - Vector Search:
    - FAISS
- Database:
  - MongoDB
  - Cashes Questions for faster retrieval
  - Locally hosted
- Interface:
  - Streamlit
  - Simple UI
  - Locally hosted

### [See PowerPoint Presentation][pptx]

___

### Technologies used:

- Ubuntu
- RTX 4050 & CUDA
- Pycharm
- Jupyter Notebook

[pptx]:Finance_RAG.pptx