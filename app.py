
import os
import requests
import gradio as gr
import chromadb
from bs4 import BeautifulSoup
from google import genai

# 1. Read the API key from Render
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not configured.")

gemini_client = genai.Client(api_key=api_key)

# 2. Create an in-memory vector database
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="aws_support_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# 3. AWS documentation sources
AWS_URLS = [
    "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
    "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
    "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html",
]


# 4. Generate embeddings using Gemini
def embed_texts(texts):
    embeddings = []

    # Process small batches to limit memory usage
    for start in range(0, len(texts), 8):
        batch = texts[start:start + 8]

        response = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch
        )

        embeddings.extend(
            [item.values for item in response.embeddings]
        )

    return embeddings


# 5. Build the AWS knowledge base
def build_knowledge_base():
    documents = []
    metadatas = []
    ids = []

    for url in AWS_URLS:
        try:
            response = requests.get(
                url,
                timeout=20,
                headers={
                    "User-Agent": "AWSCloudSupportAssistant/1.0"
                }
            )
            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            for element in soup(
                ["script", "style", "nav", "footer", "header"]
            ):
                element.decompose()

            text = " ".join(soup.stripped_strings)

            chunk_size = 900
            overlap = 150

            for start in range(
                0, len(text), chunk_size - overlap
            ):
                chunk = text[start:start + chunk_size]

                if len(chunk.strip()) < 100:
                    continue

                documents.append(chunk)
                metadatas.append({"url": url})
                ids.append(f"doc{len(ids)}")

        except Exception as e:
            print(f"Could not load {url}: {e}")

    if not documents:
        raise RuntimeError(
            "Could not download AWS documentation. "
            "Please retry later."
        )

    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(
        f"Loaded {len(documents)} AWS documentation chunks."
    )


build_knowledge_base()


# 6. RAG question-answering function
def ask_aws_assistant(question, history):
    if not question or not question.strip():
        return "Please enter an AWS question."

    query_embedding = embed_texts([question])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    if not results["documents"] or not results["documents"][0]:
        return (
            "I couldn't find sufficiently relevant information "
            "in the available AWS documentation."
        )

    distances = results["distances"][0]

    # Reject weak matches; this threshold may need tuning
    if not distances or min(distances) >= 0.65:
        return (
            "I couldn't find sufficiently relevant information "
            "in the available AWS documentation. "
            "Please ask about Amazon S3, Amazon EC2, or AWS IAM."
        )

    context_parts = []
    sources = []

    for document, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        context_parts.append(document)

        url = metadata.get("url")
        if url and url not in sources:
            sources.append(url)

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AWS Cloud Support Assistant.

Answer using only the AWS documentation context below.
If the context does not contain enough information, say so.
Do not invent AWS configuration details.

If the question is unrelated to AWS or the supplied
documentation, politely explain that you can only answer
questions supported by the available AWS documentation.

Context:
{context}

Question:
{question}

Provide a clear, beginner-friendly answer.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    answer = response.text or "No answer was generated."

    if sources:
        answer += "\n\n**Sources:**\n" + "\n".join(
            f"- {url}" for url in sources
        )

    return answer


# 7. Gradio web interface
demo = gr.ChatInterface(
    fn=ask_aws_assistant,
    title="AWS Cloud Support Assistant",
    description=(
        "Ask questions about Amazon S3, Amazon EC2, and AWS IAM. "
        "Answers use retrieved AWS documentation and include sources."
    ),
    examples=[
        "What is Amazon S3 used for?",
        "What is an Amazon EC2 instance?",
        "What is AWS IAM?",
        "How can I store files in Amazon S3?"
    ]
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
