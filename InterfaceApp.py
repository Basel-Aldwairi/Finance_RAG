import streamlit as st
from PIL import  Image


import RAG

@st.cache_resource
def load_rag():
    return RAG.HousingBankRAG()

result = None

image = Image.open('hbtf_logo.jpg')
st.image(image)
rag = load_rag()
st.markdown("""
## Housing Bank RAG
___
""")

top_k = st.slider('Choose top_k',min_value=1,max_value=6)
max_new_tokens = st.selectbox('Select max_new_tokens:',[128,256,512,1024])
query_text =  st.text_input('Question:',key='q')




def get_answer(query):
    return rag.search_question(query, top_k=top_k,max_new_tokens=max_new_tokens)



if query_text:
    with st.spinner('Generating answer...'):
        result = get_answer(query_text)
    if result:
        st.success('Done')
        st.markdown('___')
        st.markdown(f"""
    {result}
    """)
        st.markdown('___')
    else:
        st.error('Choose a bigger max_new_tokens')
