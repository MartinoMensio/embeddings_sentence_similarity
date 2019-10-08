import streamlit as st
import spacy
from spacy import displacy
import io
from cairosvg import svg2png

st.title('Embeddings experiment')

#@st.cache
def load_model(model_id):
    return spacy.load(model_id)

model_choice = st.radio('Model', (
    'en_core_web_sm', 'en_core_web_md', 'en_core_web_lg', 'en_vectors_web_lg', 'en_trf_bertbaseuncased_lg', 'en_trf_robertabase_lg', 'en_trf_distilbertbaseuncased_lg', 'en_trf_xlnetbasecased_lg'
    ))

with st.spinner('Loading model...'):
    nlp = load_model(model_choice)

sentence1 = st.text_input('First sentence', 'This is the first sentence')
sentence2 = st.text_input('Second sentence', 'This is the second sentence')

with st.spinner('Analysing sentences...'):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

# st.write([{
#     'text': t.text,
#     'pos': t.pos_,
#     'has_vector': t.has_vector
# } for t in doc1])

def create_svg(doc):
    svg_str = displacy.render(doc, style='dep')
    mock_file = io.BytesIO()
    svg2png(svg_str, write_to=mock_file)
    return mock_file

show_dep = st.checkbox('Show DEP trees', False)
if show_dep:
    st.image(create_svg(doc1), format='svg')
    st.image(create_svg(doc2), format='svg')

markdown = displacy.render([doc1, doc2], style='ent')
st.markdown(markdown, unsafe_allow_html=True)

st.write('document similarity', doc1.similarity(doc2))