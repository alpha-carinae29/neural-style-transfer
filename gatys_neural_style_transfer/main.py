import cv2
import numpy as np
import tensorflow as tf


def load_and_preprocess_image(img_path, w, h):
    img = cv2.imread(img_path)
    if img is None:
        raise OSError("provided image path: {} is not a valid path for image".format(img_path))
    img = img.astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    img = np.expand_dims(img, axis=0)
    return img


def build_model(model_name="vgg19", content_layer="block1_conv1", style_layers=["block1_conv1", "block2_conv1"]):
    if model_name == "vgg19":
        model = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    else:
        raise ValueError("not supported provided network: {}".format(model_name))
    content_net = tf.keras.Model(inputs=model.inputs,
                                 outputs=[layer.output for layer in model.layers if layer.name in content_layer])
    style_net = tf.keras.Model(inputs=model.inputs,
                               outputs=[layer.output for layer in model.layers if layer.name in style_layers])
    return content_net, style_net


def get_feature(model, content_image):
    preprocessed_content = tf.keras.applications.vgg19.preprocess_input(content_image)
    return model(preprocessed_content)


def compute_loss(content_feature, style_feature):
    pass


def apply_optimization_step(loss):
    pass


def initilize_image(method, content_img, style_img):
    if method == "content":
        return content_img
    if method == "style":
        return style_img
    if method == "random":
        return np.random.rand(*content_img.shape)


def log():
    pass


def stylize():
    pass


if __name__ == "__main__":
    content_img_path = "/mnt/DATA/neural-style-transfer/images/taj_mahal.jpg"
    style_img_path = "/mnt/DATA/neural-style-transfer/images/starry-night.jpg"
    w, h = 224, 224
    content_img = load_and_preprocess_image(content_img_path, w, h)
    style_img = load_and_preprocess_image(style_img_path, w, h)
    content_model, style_model = build_model("vgg19")
    content_feature = get_feature(content_model, content_img)
    style_feature = get_feature(style_model, style_img)
    print("finished!")
