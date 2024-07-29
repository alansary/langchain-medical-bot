from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import os

# Set the environment variable to allow dangerous deserialization
os.environ['ALLOW_DANGEROUS_DESERIALIZATION'] = "true"

DB_FAISS_PATH = 'vectorstore/db_faiss'

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )

    return prompt

# Retrieval QA Chain
# The default chain_type="stuff" uses ALL of the text from the documents in the prompt.
# map_reduce: It separates texts into batches (as an example, you can define batch size in llm=OpenAI(batch_size=5)), feeds each batch with the question to LLM separately, and comes up with the final answer based on the answers from each batch.
# refine : It separates texts into batches, feeds the first batch to LLM, and feeds the answer and the second batch to LLM. It refines the answer by going through all the batches.
# map-rerank: It separates texts into batches, feeds each batch to LLM, returns a score of how fully it answers the question, and comes up with the final answer based on the high-scored answers from each batch.
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        # model="TheBloke/Llama-2-7B-Chat-GGML",
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )

    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})

    return response

# chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    ###################################################################################
    # You can visualize the intermediate steps by clicking on the dropdown for any step
    ###################################################################################
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
