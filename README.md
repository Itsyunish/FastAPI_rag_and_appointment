# RAG FastAPI Project

This project is a **Retrieval-Augmented Generation (RAG) API** built with FastAPI, using **LangChain**, **MongoDB**, **Redis**, and **Pinecone** for document storage, embeddings, and conversation history.

---

## Features

- Upload and process PDF/TXT files.
- Split documents using **RecursiveCharacter** or **Character** chunking.
- Generate embeddings with **HuggingFace sentence transformers**.
- Store vectors in **Pinecone** for semantic search.
- Retrieve answers using **ChatGoogleGenerativeAI**.
- Store conversation history in **Redis**.
- Save file metadata and appointments in **MongoDB**.
- Endpoints for querying, conversation history, and scheduling appointments.

