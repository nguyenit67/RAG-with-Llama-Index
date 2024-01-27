import os
import sys

os.environ["OPENAI_API_KEY"] = "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"

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
import gradio as gr

from llama_index.llms import OpenAI


# Function to get the answer using your existing logic
def get_answer(question):
    # Load documents from the "data" directory
    documents = SimpleDirectoryReader("data").load_data()

    # Create a VectorStoreIndex from the documents
    index = VectorStoreIndex.from_documents(documents)

    # Create a retriever from the index
    retriever = index.as_retriever(similarity_top_k=3)

    # Retrieve relevant contexts for the question
    contexts = retriever.retrieve(question)
    context_list = [n.get_content() for n in contexts]

    # Initialize OpenAI language model
    llm = OpenAI(model="gpt-3.5-turbo")

    # Create a prompt by combining the retrieved contexts and the question
    prompt = "\n\n".join(context_list + [question])

    # Get the response from the language model
    response = llm.complete(prompt)

    return response

# Create a Gradio interface
iface = gr.Interface(fn=get_answer, inputs="text", outputs="text")

# Launch the interface
iface.launch()