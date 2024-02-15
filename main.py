import os 
from constants import open_ai_key
import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
#settinf the Environment for OpwnAi
os.environ["OPENAI_API_KEY"]  = open_ai_key
#preparing the UI
st.title("DSA Search GPT ")
input_text = st.text_input("Search")
#Prompt template
prompt_input = PromptTemplate(
    input_variables = ['TOPIC'],
    template = "Tell me about {TOPIC}"
)
#initialising the language model
llm = OpenAI(temperature=0.8)
#Memory: There are different types of memory in Lnagchain, like conversation buffer memory, in ehich we can pass the input and output as a conversation to nect chain 
Topic_memory = ConversationBufferMemory(input_key='TOPIC',memory_key='chat_history')
Use_case_memory = ConversationBufferMemory(input_key = 'Topic', memory_key = 'chat_history')

#Initialising the LLM chain as , LLMchains means Havinh an interlink between difference LLMS

chain1 = LLMChain(llm=llm, prompt=prompt_input,verbose=True,output_key='Topic',memory = Topic_memory) 
prompt_input_2 = PromptTemplate(
    input_variables = ['Topic'],
    template = "Real - Life use Case of  {Topic}"
)
chain2 = LLMChain(llm = llm, prompt = prompt_input_2,verbose = True, output_key='Real-life-use-case',memory = Use_case_memory)
#we can run the chains one by one, or we can also run the chhain sequentially all at once
Parent_chain = SequentialChain(chains = [chain1,chain2],verbose = True,input_variables=['TOPIC'],output_variables=['Topic','Real-life-use-case'])



#when the use guves the input now we will start interacting with the llm model
if input_text:
    st.write(Parent_chain({'TOPIC':input_text}))

    with st.expander(f"{input_text} Introduction:"):
        st.info(Topic_memory.buffer)
    with st.expander("Real Life Examples:"):
        st.info(Use_case_memory.buffer)


