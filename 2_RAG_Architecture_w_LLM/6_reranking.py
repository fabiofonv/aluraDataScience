# %%
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(verbose=True)
# %%
# Loaders and chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings and LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# VectorStore
from langchain_community.vectorstores import Chroma

# Prompt
from langchain_core.prompts import PromptTemplate
# %%
PERSIST_DIRECTORY = "./chroma_rh"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
# %%
@st.cache_data
def load_docs(paths):
    """
    Loads and processes documents from a list of file paths using PyPDFLoader.

    This function iterates through the provided paths, uses PyPDFLoader to
    load the PDF files, and processes their metadata to add a reference
    to the originating file path. The combined list of processed documents
    is returned.

    :param paths: List of relative file paths to the PDF documents to be loaded.
    :type paths: list[str]
    :return: A list of processed documents with metadata updated to include
        their originating file paths.
    :rtype: list
    """
    # paths = [
    #     "política_ferias.pdf",
    #     "política_home_office.pdf",
    #     "codigo_conduta.pdf"
    # ]

    documents = []

    for path in paths:
        loader = PyPDFLoader(f"../data/{path}")
        docs = loader.load()

        for doc in docs:
            doc.metadata["document"] = f"../data/{path}"

        documents.extend(docs)

    return documents

def generate_chunks(documents):
    """
    Generate chunks from the provided documents using a recursive character text splitter.
    This function takes a collection of input documents and splits them into smaller
    chunks of text based on the specified chunk size and chunk overlap. The purpose
    of this is typically to preprocess text data for further processing, such as
    training machine learning models or feeding data into algorithms.

    :param documents: A list of documents to be split into smaller chunks. Each document
        is expected to be a text input that the splitter will segment into smaller parts.
        The input should follow the format required by the text splitting library.
    :type documents: list
    :return: A list of document chunks generated from the input documents. Each resulting
        chunk is a substring of one of the input documents, processed according to
        the splitting rules (chunk size and overlap).
    :rtype: list
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)

def enrich_chunks(chunks):
    """
    Analyzes and categorizes text chunks based on specific keywords and updates metadata
    with the corresponding category.

    :param chunks: A list of chunks, where each chunk is expected to have ``page_content``
        containing text and ``metadata`` for storing the category information.
    :type chunks: list
    :return: The input list of chunks with updated ``metadata`` fields for each chunk to
        include the detected category.
    :rtype: list
    """
    for chunk in chunks:
        text = chunk.page_content.lower()

        if "férias" in text:
            chunk.metadata["category"] = "vacation"
        elif "home office" in text:
            chunk.metadata["category"] = "home_office"
        elif ("conduta" in text) or ("ética" in text):
            chunk.metadata["category"] = "conduct"
        else:
            chunk.metadata["category"] = "general"

    return chunks
# %%
@st.cache_resource
def create_vectorstore(_chunks):
    """
    Creates and initializes a vector store using the provided document chunks.

    The function uses embeddings based on the Google Generative AI model and utilizes
    the Chroma vector database to store the embedded document chunks. The resulting
    vector store is persisted in the specified directory.

    :param _chunks: A list of document chunks to be embedded and stored.
                    Each chunk represents a portion of text or data to be processed.
    :return: Initialized and persisted vector store for the given document chunks.
    :rtype: Chroma
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    return vectorstore
# %%
def rerank_documents(question, documents, llm):
    """
    Reranks a list of documents based on their relevance to a given question using a language model.

    This function evaluates the relevance of each document in the list based on the user's question.
    A language model is used to assign scores to each document, indicating its relevance. Documents
    are then sorted in descending order based on their assigned scores, with the most relevant ones
    appearing first in the returned list.

    :param question: The question posed by the user, for which the documents need to be evaluated.
    :type question: str
    :param documents: A list of documents to evaluate. Each document must expose its content via the
        `page_content` attribute.
    :type documents: List[Document]
    :param llm: An instance of a language model capable of invoking prompts and providing
        outputs.
    :type llm: LanguageModel
    :return: A list of documents sorted by their relevance to the user's question, from most relevant
        to least relevant.
    :rtype: List[Document]
    """
    prompt_rerank = PromptTemplate(
        input_variables=["question", "text"],
        template="""
        Você é um especialista em políticas internas de RH.
        Pergunta do usuário: {question}
        Trecho do documento: {text}
        Avalie a relevância desse trecho para responder a pergunta.
        Responda apenas com um número de 0 a 10.
        """
    )

    documents_w_score = []

    for doc in documents:
        score = llm.invoke(
            prompt_rerank.format(
                question=question,
                text=doc.page_content
            )
        ).content

        try:
            score = float(score)
        except:
            score = 0

        documents_w_score.append((score, doc))

    sorted_docs = sorted(
        documents_w_score,
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in sorted_docs]
# %%
def answer_question(question, vectorstore):
    """
    Answers a question based on provided context retrieved from a vector store. The process includes similarity
    search for relevant documents, reranking of retrieved documents, and generating a response using a language
    model. The final response is tailored to simulate a corporate HR agent answering based only on internal
    policies.

    :param question: The question or query to answer.
    :type question: str
    :param vectorstore: A vector store containing indexed documents to search for context.
    :type vectorstore: Any
    :return: A tuple containing the generated answer and the final context used for generating the answer.
    :rtype: tuple
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0
    )

    retrieved_docs = vectorstore.similarity_search(
        question,
        k=8
    )

    reranked_docs = rerank_documents(
        question,
        retrieved_docs,
        llm
    )

    final_context = reranked_docs[:4]

    text_context = "\n\n".join(
        [doc.page_content for doc in final_context]
    )

    final_prompt = f"""
    Você é um agente de RH corporativo.
    Responda APENAS com base nas políticas internas abaixo.
    Contexto:
    {text_context}
    Pergunta:
    {question}
    """

    answer = llm.invoke(final_prompt)

    return answer.content, final_context
# %%
st.set_page_config(page_title="Agente de RH com RAG", layout="wide")
st.title("Chat de RG - Políticas Internas")

question = st.text_input("Digite sua pergunta sobre políticas internas de RH:")

paths = [
        "politica_ferias.pdf",
        "politica_home_office.pdf",
        "codigo_conduta.pdf"
]

if question:
    with st.spinner("Consultando políticas internas..."):
        documents = load_docs(paths)
        chunks = generate_chunks(documents)
        chunks = enrich_chunks(chunks)
        vectorstore = create_vectorstore(chunks)

        answer, source = answer_question(question, vectorstore)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Source")
    for i, doc in enumerate(source, start=1):
        st.markdown(f"**Trecho {i}**")
        st.write(f"Documento: {doc.metadata.get('document')}")
        st.write(f"Categoria: {doc.metadata.get('category')}")
        st.write(doc.page_content)
        st.divider()