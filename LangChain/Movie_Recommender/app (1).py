import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

def langchain_helper():
  llm = OpenAI(temperature=0.7)
  name_prompt = PromptTemplate(
    input_variables=['movie_name_given'],
    template= "Recommend one movie based on this {movie_name_given}.Provide only name."
    )
  name_chain = LLMChain(llm=llm,prompt=name_prompt,output_key='movie_name')
    
  llm = OpenAI(temperature=0.7)
  lead_prompt = PromptTemplate(
    input_variables=['movie_name'],
    template= "Give me names of the actors in that {movie_name}."
    )
  lead_chain = LLMChain(llm=llm,prompt=lead_prompt,output_key='lead_name')
    
  llm = OpenAI(temperature=0.7)
  overview_prompt = PromptTemplate(
    input_variables=['movie_name'],
    template= "Give me an overview of that {movie_name} in 100 words."
    )
  overview_chain = LLMChain(llm=llm,prompt=overview_prompt,output_key='overview')
  input_vars = ['movie_name_given']
  output_vars=['movie_name','lead_name','overview']
  return name_chain, lead_chain, overview_chain,input_vars,output_vars

def generate_response(name):
  name_movie, lead, overview, input_vars, output_vars = langchain_helper()
  chain = SequentialChain(
    chains=[name_movie,lead,overview],
    input_variables=input_vars,
    output_variables=output_vars
  )
  reply = chain({input_vars[0]:name})

  return reply
st.title("ALL OUT OF MOVIES")
st.image("movie_banner.webp")
prompt = st.text_input(label="Type a name of any movie.")
if st.button("Submit"):
  info_dir = generate_response(prompt)
  st.subheader("Movie Recommended:")
  st.write(f"Movie Name : {info_dir['movie_name'].strip()}")
  st.subheader("Actors in the movie:")
  st.write(f"{info_dir['lead_name']}")
  st.subheader("Movie Overview:")
  st.write(f"{info_dir['overview']}")