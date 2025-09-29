import streamlit as st
from PIL import  Image
from streamlit import button

import RAG

@st.cache_resource
def load_rag():
    return RAG.HousingBankRAG()

if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


image = Image.open('hbtf_logo.jpg')
st.image(image)
rag = load_rag()
st.markdown("""
## Housing Bank RAG
___
""")

top_k = st.slider('Choose top_k',min_value=1,max_value=6,value= 3)
max_new_tokens = st.selectbox('Select max_new_tokens:',[128,256,512,1024])
alpha = st.slider('Choose alpha (BM25 -- FAISS):',min_value=0.0,max_value=1.0,value= 0.5,step=0.05)
if st.checkbox('Enable Database Search'):
    use_db = True
else:
    use_db = False


query_text =  st.text_input('Question:',key='q')

ask_question = 'Ask Question'
delete_question = 'Delete Question'
reset_database = 'Reset Database'
view_database = 'View Database'

# indexing = {
#     ask_question : 0,
#     delete_question : 1,
#     reset_database = 'Reset Database'
#     view_database = 'View Database'
# }

action = st.selectbox('Select Action: ',[ask_question,
                                         delete_question,
                                         reset_database,
                                         view_database])

def get_answer(query):
    return rag.search_question(query, top_k=top_k,max_new_tokens=max_new_tokens,alpha=alpha,use_db=use_db)

last_query = ''
asked = False
if st.button(f'{action}!'):
    st.session_state.last_query = None
    st.session_state.last_result = None

    match action:
        case 'Ask Question':
            with st.spinner('Generating answer...'):
                result = get_answer(query_text)
            if result:
                st.session_state.last_query = query_text
                st.session_state.last_result = result
            else:
                st.error('Choose a bigger max_new_tokens')

        case 'Delete Question':
            if rag.db.delete(rag.embed_query(query_text),deletion_type=rag.db.delete_one):
                st.success(f'Deleted: {query_text}')
            else:
                st.error(f'Unable to find: {query_text}')

        case 'Reset Database':
            if rag.db.delete(rag.embed_query(query_text),deletion_type=rag.db.delete_all):
                st.success('Reset Database')

        case 'View Database':
            data = rag.db.get_all_questions()
            if data:
                st.success('Database:')
                for i, d in enumerate(data):
                    st.write(f'{i + 1} : {d['question']}')
            else:
                st.error('Empty Database')


if st.session_state.last_result:
    st.success('Done')
    st.markdown('___')
    st.markdown(f"""
               {st.session_state.last_result}
               """)
    st.markdown('___')
    if st.button('Save answer in Database'):
        rag.db.delete(rag.embed_query(query_text), deletion_type=rag.db.delete_one)
        rag.db.insert_question(st.session_state['last_query'],
                               st.session_state['last_result'],
                               rag.embed_query(st.session_state['last_query']))
        st.session_state.last_query = None
        st.session_state.last_result = None
        st.session_state.pop('last_query')
        st.success('Saved in Database')
