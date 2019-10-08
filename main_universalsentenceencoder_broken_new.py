import tensorflow_hub as hub
import tensorflow as tf
import streamlit as st
import numpy


st.title('universal-sentence-encoder experiment')

with st.spinner('Loading USE model...'):
    imported = hub.load("https://tfhub.dev/google/universal-sentence-encoder/2")
    embed = imported.signatures['default']

sentence1 = st.text_input('First sentence', 'This is the first sentence')
sentence2 = st.text_input('Second sentence', 'This is the second sentence')

sentences = [sentence1, sentence2]

with st.spinner('Encoding...'):
    tensors = tf.convert_to_tensor(sentences)
    embeddings_tensor = embed(tensors)
    st.write(embeddings_tensor)
    st.write(embeddings_tensor['default'])
    session = tf.compat.v1.Session()
    st.write(embeddings_tensor['default'].eval(session=session))
    embeddings = embeddings_tensor['default'].ev()

similarity = numpy.dot(embeddings[0], embeddings[1]) / (numpy.sqrt(numpy.dot(embeddings[0],embeddings[0])) * numpy.sqrt(numpy.dot(embeddings[1], embeddings[1])))

st.write('similarity', similarity)

# print(embeddings)


#config = tf.compat.v1.ConfigProto()
#config.graph_options.rewrite_options.shape_optimization = 2
# session = tf.compat.v1.Session(config=config)
# embeddings_result = session.run(embeddings)
