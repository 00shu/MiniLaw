# MiniLaw
This project implements a RAG system to answer questions based on the training corpus provided. In this case the training corpus is rules and regulation pdfs provided by the government.

There are two files in this project: ingest.py and query.py

1) ingest.py is responsible for the creation of vector database and storing it.
2) query.py performs the actual RAG response creation, by using a pretrained model ( in this case: TheBloke/Llama-2-7B-Chat-GGML ) and the vector database generated using the ingest.py

This code can be used as a template for development of other RAG systems.
