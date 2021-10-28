import streamlit as st


def load_model(model_file_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_file_path)
    return model
