# import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load the model only once and cache the result
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")

def compute_sim(sent1,sent2,model):
    
    embeddings_1 = model.encode(sent1, normalize_embeddings=True)
    embeddings_2 = model.encode(sent2, normalize_embeddings=True)
    similarity = embeddings_1 @ embeddings_2.T
    return similarity

loaded_model = load_model()

sentence_1 = st.text_input('Please enter the first sentence')
sentence_2 = st.text_input('Please enter the second sentence')
sim_score = compute_sim(sentence_1,sentence_2,loaded_model)
if st.button('Submit'):
    st.write("Finally the similarity score is computed!!!......")
    if sim_score > 0.5:
        st.title(f'They are :green[Similar], score:{sim_score}')
    else:
        st.title(f'They are :red[Different], score: {sim_score}')
else:
    st.stop()