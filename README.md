
# AWS Cloud Support Assistant

A Generative AI-powered assistant that answers AWS cloud support questions using Retrieval-Augmented Generation (RAG), Gemini embeddings, and official AWS documentation.

## Live Demo

Try the application here: **[AWS Cloud Support Assistant]((https://aws-cloud-support-assistant.onrender.com/))**

## Features
- Retrieves relevant information from AWS documentation.
- Uses Gemini embeddings for semantic search.
- Generates beginner-friendly answers using Gemini.
- Includes source links to AWS documentation.
- Filters questions that are unrelated to the available documentation.

## Technologies Used
- Python
- Google Gemini API
- ChromaDB
- Gradio
- Retrieval-Augmented Generation (RAG)
- BeautifulSoup
- Render

## Supported Topics
- Amazon S3
- Amazon EC2
- AWS Identity and Access Management (IAM)

# AWS Cloud Support Assistant using RAG

An AI-powered AWS Cloud Support Assistant that answers questions about Amazon S3, Amazon EC2, and AWS Identity and Access Management (IAM) using Retrieval-Augmented Generation (RAG).

## Project Overview

This project combines semantic search with a large language model to provide beginner-friendly answers grounded in AWS documentation. It retrieves relevant document chunks before generating an answer and returns source links so users can verify the information.

## Architecture

1. **Knowledge source:** AWS documentation for Amazon S3, Amazon EC2, and AWS IAM.
2. **Text processing:** Documentation is divided into smaller chunks.
3. **Embedding generation:** Sentence Transformers (`all-MiniLM-L6-v2`) converts text and questions into embeddings.
4. **Vector database:** ChromaDB retrieves relevant documentation chunks using semantic similarity.
5. **Relevance filtering:** A distance threshold filters out insufficiently relevant results.
6. **Answer generation:** Gemini generates answers using the retrieved context.
7. **User interface:** Gradio provides an interactive chat interface with source links.

## Technologies Used

- Python
- Google Colab
- Google Gemini API
- Sentence Transformers
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- Gradio
- AWS documentation

## Features

- Answers questions about S3, EC2, and IAM.
- Retrieves relevant AWS documentation before answering.
- Provides documentation source links.
- Rejects questions when sufficiently relevant documentation is not found.
- Includes a browser-based chat interface.
- Includes a basic evaluation report.

## Evaluation

The initial evaluation used five test questions:

| Test category | Questions | Result |
|---|---:|---|
| Amazon S3 | 2 | Passed |
| Amazon EC2 | 1 | Passed |
| AWS IAM | 1 | Passed |
| Unrelated question | 1 | Passed |
| **Total** | **5** | **5 passed** |

**Initial test pass rate: 100% (5/5).**

This is a small functional test, not a comprehensive measure of answer accuracy. Further evaluation is needed to assess retrieval quality, factual correctness, and source-grounded answers across a larger test set.

## Project Files

- `AWS_Cloud_Support_Assistant.ipynb` — Google Colab notebook containing the implementation.
- `aws_rag_evaluation.csv` — results of the initial evaluation.
- `README.md` — project documentation.

## Setup

1. Open the notebook in Google Colab.
2. Install the required Python packages listed in the notebook.
3. Add your Gemini API key to Colab Secrets using the name `GEMINI_API_KEY`.
4. Run the notebook cells in order to load the documentation, create embeddings, initialize ChromaDB, and launch the Gradio interface.

Do not hardcode API keys or tokens in the notebook or commit them to GitHub.

## Limitations

- The knowledge base currently covers selected documentation for S3, EC2, and IAM.
- Answers depend on the retrieved context and the quality of the source documents.
- The retrieval distance threshold requires further validation.
- The initial evaluation contains only five questions.

## Future Improvements

- Add documentation for more AWS services.
- Expand the evaluation dataset.
- Measure retrieval precision and answer faithfulness.
- Improve error handling and logging.
- Deploy the application to a persistent hosting platform.

## Disclaimer

This is an educational project and is not an official AWS support service. Always verify important technical decisions against the official AWS documentation.
