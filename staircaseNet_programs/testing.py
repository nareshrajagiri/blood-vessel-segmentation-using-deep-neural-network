import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import auc, specificity, sensitivity

H = 512         # Height
W = 512         # Width

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)
    return ori_x, x

def load_data(path, extension):
    files = sorted(glob(os.path.join(path, f"*.{extension}")))
    return files

def save_results(ori_x, ori_y, ori_z, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_z, line, ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    create_dir("/content/drive/MyDrive/mini_project/files/results")

    with CustomObjectScope({'sensitivity': sensitivity, 'specificity': specificity, 'auc': auc}):
        model = tf.keras.models.load_model('/content/drive/MyDrive/mini_project/output_storage/model.h5')

    test_x = load_data("/content/drive/MyDrive/mini_project/a_data/test_preproccessing", "tiff")
    test_y = load_data("/content/drive/MyDrive/mini_project/a_data/test/mask", "jpg")
    coloured_images = load_data("/content/drive/MyDrive/mini_project/a_data/test/image", "jpg")

    if not test_x or not test_y or not coloured_images:
        print("No data to calculate metrics.")
        exit()

    SCORE = []
    for x, y, z in tqdm(zip(test_x, test_y, coloured_images), total=len(test_x)):
        name = os.path.basename(x).split(".")[0]

        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)
        ori_z, z = read_image(z)

        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)
        y_pred = np.squeeze(y_pred, axis=-1)

        save_image_path = os.path.join("/content/drive/MyDrive/mini_project/files/results", name + ".png")
        save_results(ori_x, ori_y, ori_z, y_pred, save_image_path)

        y = y.flatten()
        y_pred = y_pred.flatten()

        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "Acc", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("/content/drive/MyDrive/mini_project/files/score.csv")
