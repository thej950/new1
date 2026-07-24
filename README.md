# Enterprise Agentic AI Platform

## Project Overview

The Enterprise Agentic AI Platform is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions. The platform retrieves relevant information from uploaded documents and generates context-aware responses using a Large Language Model (LLM). It follows a modular architecture that can be easily extended for enterprise use cases.

---

# Project Modules

## 1. FastAPI

FastAPI is the backend framework used to build REST APIs for the project. It manages request handling, validation, routing, and API documentation through Swagger UI.

---

## 2. Upload Service

The Upload Service allows users to upload PDF documents into the application. It validates the uploaded files, generates a unique document ID, and stores them for further processing.

---

## 3. PDF Processor

The PDF Processor extracts text content from uploaded PDF documents using PyMuPDF. The extracted text becomes the input for the RAG pipeline.

---

## 4. Text Chunking

The Chunking module divides large documents into smaller overlapping text chunks. This improves retrieval accuracy by allowing semantic search on manageable pieces of text.

---

## 5. Embedding Service

The Embedding Service converts each text chunk into a numerical vector using the SentenceTransformer model. These embeddings capture the semantic meaning of the document.

---

## 6. Vector Store (FAISS)

FAISS stores the generated embeddings and performs efficient similarity searches. It retrieves the most relevant document chunks for a user's question.

---

## 7. Retrieval Service

The Retrieval Service converts the user's question into an embedding and searches the FAISS index. It returns the top matching document chunks as contextual information.

---

## 8. Prompt Builder

The Prompt Builder combines the retrieved document context with the user's question. It creates a structured prompt that is sent to the language model for answer generation.

---

## 9. LLM Service

The LLM Service acts as an abstraction layer between the application and the language model. It supports both Mock LLM for local testing and Amazon Bedrock for production deployment.

---

## 10. Mock LLM

The Mock LLM simulates AI responses during local development when Amazon Bedrock is unavailable. This enables testing without requiring cloud resources.

---

## 11. Amazon Bedrock

Amazon Bedrock is the cloud-based LLM service used in production. It receives the generated prompt and returns AI-generated responses using foundation models.

---

## 12. Agent Orchestrator

The Agent Orchestrator is responsible for routing user questions to the most suitable AI agent. It analyzes the query and coordinates the overall response generation process.

---

## 13. HR Agent

The HR Agent specializes in answering Human Resources related questions. It retrieves HR-specific information from uploaded documents before generating responses.

---

## 14. IT Agent

The IT Agent handles technical support related questions such as VPN, passwords, software, and system access. It follows the same retrieval and response workflow.

---

## 15. Finance Agent

The Finance Agent processes finance-related questions including invoices, reimbursements, expenses, and payments. It provides responses based on retrieved document content.

---

## 16. Multi-Agent Collaboration

For complex questions covering multiple domains, the orchestrator can invoke multiple agents. Their responses are merged to generate a single comprehensive answer.

---

## 17. Configuration Management

Configuration values such as model selection, API settings, and environment variables are managed through a centralized configuration module. This makes the application easy to configure across different environments.

---

## 18. Logging

The Logging module records application events, API requests, retrieval operations, and system errors. These logs simplify debugging and monitoring during development.

---

## 19. API Endpoints

### **POST /upload**

Uploads a PDF document into the system for processing.

### **POST /embed**

Generates embeddings and updates the FAISS vector index.

### **POST /retrieve**

Retrieves the most relevant document chunks based on semantic similarity.

### **POST /ask**

Performs the standard RAG workflow to generate an answer from retrieved document context.

### **POST /agent-chat**

Routes the question through the Agent Orchestrator and generates an intelligent response using specialized agents.

---

# Overall Workflow

1. Upload a PDF document.
2. Extract text from the document.
3. Split the text into smaller chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in the FAISS vector database.
6. Receive the user's question.
7. Route the question to the appropriate AI agent.
8. Retrieve relevant document chunks.
9. Build the LLM prompt.
10. Generate the final answer using the Mock LLM or Amazon Bedrock.
11. Return the answer along with the relevant document sources.

---

