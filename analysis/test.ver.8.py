import numpy as np # https://numpy.org/ja/
import pandas as pd # https://pandas.pydata.org/
from sklearn.model_selection import train_test_split

def resize_map(map):
    from PIL import Image

    #平均で置換する

    # リサイズ後のサイズを指定
    resize_shape = (32, 32)

    # 0以外の値の平均を計算
    non_zero_average = np.mean(map[map != 0])

    # 0の値を平均値で置換
    map[map == 0] = non_zero_average

    # 1を減算
    map = Image.fromarray(map - 1.0)

    # PILイメージを作成
    resized_map = map.resize(resize_shape, Image.LANCZOS)

    return np.asarray(resized_map)


# def scale_image_imgaug(image, scale_range=(0.9, 1.1)):
#     import imgaug.augmenters as iaa

#     scale = iaa.Affine(scale={"x": scale_range, "y": scale_range})
#     scaled_image = scale.augment_image(image)
#     return scaled_image


# def add_gaussian_noise_imgaug(image):
#     import imgaug.augmenters as iaa

#     # imgaugを使用してガウシアンノイズを追加
#     # ここでノイズのスケールを0から0.1までに設定
#     augmenter = iaa.AdditiveGaussianNoise(scale=(0, 0.1))
#     noisy_image = augmenter.augment_image(image)

#     # 値を0〜1の範囲に制限
#     noisy_image = np.clip(noisy_image, 0, 1)
#     return noisy_image

# def add_salt_pepper_noise_and_clip(image, salt_pepper=(0.01, 0.05)):
#     import imgaug.augmenters as iaa

#     augmenter = iaa.SaltAndPepper(salt_pepper)
#     noisy_image = augmenter.augment_image(image)

#     # ピクセル値を0〜1の範囲に制限
#     noisy_image_clipped = np.clip(noisy_image, 0, 1)
#     return noisy_image_clipped


def preprocess_map(df, resize_map):
    # データの正規化
    preprocessed_maps = np.array([resize_map(x) for x in df['waferMap']])

    # # 1. 画像を水平方向に反転
    # flipped_horizontally = np.flip(preprocessed_maps, axis=2)
    # preprocessed_maps = np.concatenate((preprocessed_maps, flipped_horizontally), axis=0)

    # # 2. 画像を垂直方向に反転
    # flipped_vertically = np.flip(preprocessed_maps, axis=1)
    # preprocessed_maps = np.concatenate((preprocessed_maps, flipped_vertically), axis=0)

    # 3. 画像を90度回転
    rotated_90 = np.rot90(preprocessed_maps, k=1, axes=(1, 2))
    preprocessed_maps = np.concatenate((preprocessed_maps, rotated_90), axis=0)

    # 4. 画像を180度回転
    rotated_180 = np.rot90(preprocessed_maps, k=2, axes=(1, 2))
    preprocessed_maps = np.concatenate((preprocessed_maps, rotated_180), axis=0)

    # 5. 画像を270度回転
    rotated_270 = np.rot90(preprocessed_maps, k=3, axes=(1, 2))
    preprocessed_maps = np.concatenate((preprocessed_maps, rotated_270), axis=0)

    # # 6. 画像のスケールを変更
    # scaled_maps = np.array([scale_image_imgaug(x) for x in preprocessed_maps])
    # preprocessed_maps = np.concatenate((preprocessed_maps, scaled_maps), axis=0)

    # # 7. ノイズを追加
    # noisy_maps = np.array([add_gaussian_noise_imgaug(x) for x in preprocessed_maps])
    # preprocessed_maps = np.concatenate((preprocessed_maps, noisy_maps), axis=0)

    # # 8. 塩胡椒ノイズの追加
    # augmented_maps = np.array([augment_image_imgaug(x) for x in preprocessed_maps])
    # preprocessed_maps = np.concatenate((preprocessed_maps, augmented_maps), axis=0)

    # データの形状を変更
    preprocessed_maps = preprocessed_maps.reshape(preprocessed_maps.shape + (1,))

    return preprocessed_maps


def initialize_cnn(input_shape, failure_types_classes):
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        # 畳み込みブロック1
        tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same', input_shape=(input_shape)),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        # 畳み込みブロック2
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        # 畳み込みブロック3
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        # ブロック4
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0),

        # 出力層
        tf.keras.layers.Dense(failure_types_classes),
    ])

    return model


def calculate_class_weights(train_labels):
    from sklearn.utils.class_weight import compute_class_weight
    # クラスの重みを計算
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(train_labels), 
                                         y=train_labels)
    # クラスの重みを辞書型に変換
    return dict(enumerate(class_weights))


def solution(x_test_df, train_df):
    import os
    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    failure_types = list(train_df['failureType'].unique())

    train_maps = preprocess_map(train_df, resize_map)
    train_labels = np.array([failure_types.index(x) for x in train_df['failureType']] * 8)

    failure_types_classes = len(failure_types)
    input_shape = train_maps[0].shape

    # クラスの重みを計算
    class_weights = calculate_class_weights(train_labels)

    # モデルの作成
    model = initialize_cnn(input_shape, failure_types_classes)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(train_maps, train_labels, epochs=10, class_weight=class_weights)

    test_maps = preprocess_map(x_test_df, resize_map)
    map_classes = len(x_test_df['waferMap'])

    # 予測の統合
    test_predictions = model.predict(test_maps)
    aggregated_logits = np.zeros((map_classes, failure_types_classes), dtype=np.float64)
    for n in range(len(test_predictions) // map_classes):
        aggregated_logits += test_predictions[map_classes * n : map_classes * (n + 1)]
    
    predictions = tf.nn.softmax(aggregated_logits).numpy()
    answer = [failure_types[x.argmax()] for x in predictions]

    return pd.DataFrame({'failureType': answer}, index=x_test_df.index)

def plot_confusion_matrix_and_accuracy(y_true, y_pred, classes):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Confusion matrixの計算
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # ヒートマップとしてプロット
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # 各クラスごとの正確さと最も間違えやすいクラスを表示
    print("\nClass Accuracy and Most Common Errors:")
    for i, class_name in enumerate(classes):
        accuracy = cm[i, i] / cm[i, :].sum()
        print(f"{class_name}: Accuracy: {accuracy * 100:.2f}%")
        
        # 最も間違えやすいクラスを特定
        error_indices = cm[i, :].argsort()[-2:-1] if accuracy < 1 else []
        for error_index in error_indices:
            error_rate = cm[i, error_index] / cm[i, :].sum()
            error_class = classes[error_index]
            print(f"    Most common error: Mistaken for {error_class} ({error_rate * 100:.2f}%)")

# データのインポート
df=pd.read_pickle("../input/LSWMD_25519.pkl")

# テスト用と学習用のデータを作成（テストする際は、random_stateの値などを編集してみてください）
train_df, test_df = train_test_split(df, stratify=df['failureType'], test_size=0.10, random_state=42)

y_test_df = test_df[['failureType']]
x_test_df = test_df.drop(columns=['failureType'])

# solution関数を実行
user_result_df = solution(x_test_df, train_df)
plot_confusion_matrix_and_accuracy(y_test_df['failureType'], user_result_df['failureType'], df['failureType'].unique())

average_accuracy = 0
# ユーザーの提出物のフォーマット確認
if type(y_test_df) == type(user_result_df) and y_test_df.shape == user_result_df.shape:
    # 平均精度の計算
    accuracies = {}
    for failure_type in df['failureType'].unique():
        y_test_df_by_failure_type = y_test_df[y_test_df['failureType'] == failure_type]
        user_result_df_by_failure_type = user_result_df[y_test_df['failureType'] == failure_type]
        matching_rows = (y_test_df_by_failure_type == user_result_df_by_failure_type).all(axis=1).sum()
        accuracies[failure_type] = (matching_rows/(len(y_test_df_by_failure_type)))
    
    average_accuracy = sum(accuracies.values())/len(accuracies)
print(f"平均精度：{average_accuracy*100:.2f}%")