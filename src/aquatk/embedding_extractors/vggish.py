import numpy as np
from .models.vggish import vggish_input, vggish_params, vggish_postprocess, vggish_slim, mel_features
import tensorflow.compat.v1 as tf
import six
import soundfile
from .extractor import Extractor
from tqdm import tqdm
from joblib import Parallel, delayed
import os


class VGGish(Extractor):
    def __init__(self, verbose=True, checkpoint_path=None, pca_params_path=None):
        super(VGGish, self).__init__()

        # get relative path to the checkpoint and pca_params. Theyre in ./models/
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if checkpoint_path is None:
            self.checkpoint = os.path.join(root_dir, 'models/vggish/vggish_model.ckpt')
        else: 
            self.checkpoint = checkpoint_path
        if pca_params_path is None:            
            self.pca_params = os.path.join(root_dir, 'models/vggish/vggish_pca_params.npz')
        else:
            self.pca_params = pca_params_path

    def get_embeddings(self, x, sr=16000):
        # if x is a string, then it's a path with wav files
        embeddings = []
        if isinstance(x, str):
            import os
            audio_list = []
            audio_list = Parallel(n_jobs=4)(
                delayed(vggish_input.wavfile_to_examples)(os.path.join(x, fname)) for fname in tqdm(os.listdir(x)))
            audio_list = np.array(audio_list)
        elif isinstance(x, list):
            audio_list = np.array(x)
        elif isinstance(x, np.ndarray):
            audio_list = x
        else:
            raise AttributeError

        pproc = vggish_postprocess.Postprocessor(self.pca_params)
        writer = tf.python_io.TFRecordWriter('vggish_embeddings.tfrecord')

        with tf.Graph().as_default(), tf.Session() as sess:
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, self.checkpoint)

            def get_embedding(i):
                features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
                embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
                [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: i})
                postprocessed_batch = pproc.postprocess(embedding_batch)
                return postprocessed_batch

            # number of cores:
            num_threads = os.cpu_count()

            embeddings = Parallel(n_jobs=num_threads, prefer="threads")(
                delayed(get_embedding)(i) for i in tqdm(audio_list))
        return np.array(embeddings)

    def cleanup(self):
        # ask tensorflow to cleanup
        import tensorflow as tf
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()


if __name__ == "__main__":
    vggish = VGGish()
    print(vggish.get_embeddings("/Users/ashvala/nsynth_reference"))
    vggish.cleanup()
