import tensorflow as tf

def restore(sess, saver, savePath):
    """
    If a checkpoint exists, restores the tensorflow model from the checkpoint.
    Returns the tensorflow Saver.
    """
    if tf.train.checkpoint_exists(savePath):
        try:
            saver.restore(sess, savePath)
            print('Restored model from {} successfully'.format(savePath))
        except Exception as error:
            print('Unable to restore model from path {} with error {}'.format(savePath, error))
    else:
        print('No checkpoint exists at path {}. Training from scratch...'.format(savePath))
