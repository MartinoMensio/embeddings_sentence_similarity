import streamlit as st
import nltk
import torch
import numpy
import matplotlib.pyplot as plt
from infersent_models import InferSent

st.title('infersent experiment')

with st.spinner('Downloading NLTK tokenizer...'):
    nltk.download('punkt')


V = 2
MODEL_PATH = f'encoder/infersent{V}.pkl'
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)

with st.spinner('Loading InferSent pre-trained model..'):
    infersent.load_state_dict(torch.load(MODEL_PATH))

# set word vector path for the model
W2V_PATH = st.radio('Vector path', (
    'fastText/crawl-300d-2M.vec', 'GloVe/glove.840B.300d.txt'
    ))
infersent.set_w2v_path(W2V_PATH)

sentence1 = st.text_input('First sentence', 'This is the first sentence')
sentence2 = st.text_input('Second sentence', 'This is the second sentence')

sentences = [sentence1, sentence2]

with st.spinner('Building the vocabulary of word vectors...'):
    infersent.build_vocab(sentences, tokenize=True)


embeddings = infersent.encode(sentences, tokenize=True)

similarity = numpy.dot(embeddings[0], embeddings[1]) / (numpy.sqrt(numpy.dot(embeddings[0],embeddings[0])) * numpy.sqrt(numpy.dot(embeddings[1], embeddings[1])))

st.write('similarity', similarity)

show_importance = st.checkbox('Show words importance', False)
if show_importance:
    output, idxs = infersent.visualize(sentence1, tokenize=True)
    st.pyplot()
    plt.clf()
    output, idxs = infersent.visualize(sentence2, tokenize=True)
    st.pyplot()
    # st.bar_chart(zip(output, idxs))