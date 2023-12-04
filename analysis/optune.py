import numpy as np # https://numpy.org/ja/
import pandas as pd # https://pandas.pydata.org/
from sklearn.model_selection import train_test_split

# 必要な外部パッケージは、以下の内容を編集しインストールしてください
!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install tensorflow
!pip install Pillow
!pip install optuna

import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import optuna
from PIL import Image

def resize_map(map):
    resize_shape = (32, 32)
    non_zero_average = np.mean(map[map != 0])
    map[map == 0] = non_zero_average
    map = Image.fromarray(map - 1.0)
    resized_map = map.resize(resize_shape, Image.LANCZOS)
    return np.asarray(resized_map)

def preprocess_map(df, resize_map):
    preprocessed_maps = np.array([resize_map(x) for x in df['waferMap']])
    flipped_horizontally = np.flip(preprocessed_maps, axis=2)
    preprocessed_maps = np.concatenate((preprocessed_maps, flipped_horizontally), axis=0)
    rotated_90 = np.rot90(preprocessed_maps, k=1, axes=(1, 2))
    preprocessed_maps = np.concatenate((preprocessed_maps, rotated_90), axis=0)
    rotated_180 = np.rot90(preprocessed_maps, k=2, axes=(1, 2))
    preprocessed_maps = np.concatenate((preprocessed_maps, rotated_180), axis=0)
    preprocessed_maps = preprocessed_maps.reshape(preprocessed_maps.shape + (1,))
    return preprocessed_maps

def calculate_class_weights(train_labels):
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(train_labels), 
                                         y=train_labels)
    return dict(enumerate(class_weights))
def objective(trial, train_df, x_test_df):
    # ハイパーパラメータの範囲を設定
    conv2d_filters = trial.suggest_categorical('conv2d_filters', [8, 16, 32, 64, 128])
    dense_units = trial.suggest_categorical('dense_units', [64, 128, 256, 512, 1024])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # モデルの初期化
    def initialize_cnn(input_shape, failure_types_classes):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(conv2d_filters, 3, activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(conv2d_filters * 2, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(conv2d_filters * 4, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense_units, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(dense_units, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(dense_units, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(failure_types_classes),
        ])
        return model

    # データの前処理
    failure_types = list(train_df['failureType'].unique())
    train_maps = preprocess_map(train_df, resize_map)
    train_labels = np.array([failure_types.index(x) for x in train_df['failureType']] * 8)
    failure_types_classes = len(failure_types)
    input_shape = train_maps[0].shape
    class_weights = calculate_class_weights(train_labels)

    # モデルのコンパイル
    model = initialize_cnn(input_shape, failure_types_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])

    # モデルのトレーニング
    model.fit(train_maps, train_labels, epochs=7, class_weight=class_weights)

    # 精度の計算
    test_maps = preprocess_map(x_test_df, resize_map)
    map_classes = len(x_test_df['waferMap'])

    test_predictions = model.predict(test_maps)
    aggregated_logits = np.zeros((map_classes, failure_types_classes), dtype=np.float64)
    for n in range(len(test_predictions) // map_classes):
        aggregated_logits += test_predictions[map_classes * n : map_classes * (n + 1)]

    predictions = tf.nn.softmax(aggregated_logits).numpy()
    answer = [failure_types[x.argmax()] for x in predictions]

    # 精度の計算
    correct = sum([a == b for a, b in zip(answer, x_test_df['failureType'].values)])
    accuracy = correct / len(answer)
    return accuracy

# データのインポート
df=pd.read_pickle("~/LSWMD_25519.pkl")

# テスト用と学習用のデータを作成（テストする際は、random_stateの値などを編集してみてください）
train_df, test_df = train_test_split(df, stratify=df['failureType'], test_size=0.10, random_state=42)

y_test_df = test_df[['failureType']]
x_test_df = test_df.drop(columns=['failureType'])

# Optunaのstudyを作成し、目的関数を最適化
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, train_df, x_test_df), n_trials=100)  # トライアル数は例

print("Best trial:")
print(study.best_trial)

def solution(x_test_df, train_df):
    # Optunaで見つけた最適なパラメータを使用
    best_params = study.best_params

    def initialize_cnn(input_shape, failure_types_classes):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(best_params['conv2d_filters'], 3, activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(best_params['conv2d_filters'] * 2, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(best_params['conv2d_filters'] * 4, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(best_params['dense_units'], activation=tf.nn.relu),
            tf.keras.layers.Dropout(best_params['dropout_rate']),
            tf.keras.layers.Dense(best_params['dense_units'], activation=tf.nn.relu),
            tf.keras.layers.Dropout(best_params['dropout_rate']),
            tf.keras.layers.Dense(best_params['dense_units'], activation=tf.nn.relu),
            tf.keras.layers.Dropout(best_params['dropout_rate']),
            tf.keras.layers.Dense(failure_types_classes),
        ])
        return model

    failure_types = list(train_df['failureType'].unique())
    train_maps = preprocess_map(train_df, resize_map)
    train_labels = np.array([failure_types.index(x) for x in train_df['failureType']] * 8)
    failure_types_classes = len(failure_types)
    input_shape = train_maps[0].shape
    class_weights = calculate_class_weights(train_labels)

    model = initialize_cnn(input_shape, failure_types_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    model.compile(optimizer=optimizer, 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])

    model.fit(train_maps, train_labels, epochs=7, class_weight=class_weights)

    test_maps = preprocess_map(x_test_df, resize_map)
    predictions = model.predict(test_maps)
    answer = [failure_types[np.argmax(x)] for x in predictions]

    return pd.DataFrame({'failureType': answer}, index=x_test_df.index)

# solution関数を実行
user_result_df = solution(x_test_df, train_df)

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