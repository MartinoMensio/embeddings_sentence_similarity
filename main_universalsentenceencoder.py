import tensorflow_hub as hub
import tensorflow as tf

imported = hub.load("https://tfhub.dev/google/universal-sentence-encoder/2")
embed = imported.signatures['default']
strings = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"]
tensors = tf.convert_to_tensor(strings)
embeddings = embed(tensors)
print(embeddings)


#config = tf.compat.v1.ConfigProto()
#config.graph_options.rewrite_options.shape_optimization = 2
# session = tf.compat.v1.Session(config=config)
# embeddings_result = session.run(embeddings)
