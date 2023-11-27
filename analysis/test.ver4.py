import numpy as np # https://numpy.org/ja/
import pandas as pd # https://pandas.pydata.org/
from sklearn.model_selection import train_test_split

def normalize_map(map):
    from PIL import Image

    # リサイズ後のサイズを指定
    resize_shape = (32, 32)
    
    # マップの次元を取得
    len_y, len_x = map.shape

    # マップの中心y座標とx座標を取得
    y_add = len_y // 2 + len_y % 2
    x_add = len_x // 2 + len_x % 2

    # 0の値のインデックスを取得
    y_indices, x_indices = np.where(map == 0)
    # 他の位置に置換
    map[y_indices, x_indices] = map[(y_indices + y_add) % len_y, (x_indices + x_add) % len_x]
    # 0の値のインデックスを取得
    y_indices, x_indices = np.where(map == 0)
    # 0の値を置換
    map[y_indices, x_indices] = 1

    # リサイズし1を減算
    map = Image.fromarray(map - 1.0)
    # mapから1を減算してPILイメージを作成
    resized_map = map.resize(resize_shape, Image.LANCZOS)

    return np.asarray(resized_map)


def preprocess_map(df, normalize_map):
    # データの正規化
    train_maps = np.array([normalize_map(x) for x in df['waferMap']])

    # # 1. 画像を水平方向に反転
    # flipped_horizontally = np.flip(train_maps, axis=2)
    # train_maps = np.concatenate((train_maps, flipped_horizontally), axis=0)

    # # 2. 画像を垂直方向に反転
    # flipped_vertically = np.flip(train_maps, axis=1)
    # train_maps = np.concatenate((train_maps, flipped_vertically), axis=0)

    # 3. 画像を90度回転
    rotated_90 = np.rot90(train_maps, k=1, axes=(1, 2))
    train_maps = np.concatenate((train_maps, rotated_90), axis=0)

    # 4. 画像を180度回転
    rotated_180 = np.rot90(train_maps, k=2, axes=(1, 2))
    train_maps = np.concatenate((train_maps, rotated_180), axis=0)

    # 5. 画像を270度回転
    rotated_270 = np.rot90(train_maps, k=3, axes=(1, 2))
    train_maps = np.concatenate((train_maps, rotated_270), axis=0)

    # データの形状を変更
    train_maps = train_maps.reshape(train_maps.shape + (1,))

    return train_maps


def normalize_map2(map):
    from PIL import Image

    # リサイズ後のサイズを指定
    resize_shape = (32, 32)
    
    # マップの次元を取得
    len_y, len_x = map.shape

    # マップの中心y座標とx座標を取得
    y_add = len_y // 2 + len_y % 2
    x_add = len_x // 2 + len_x % 2

    # 0の値のインデックスを取得
    y_indices, x_indices = np.where(map == 0)
    # 他の位置に置換
    map[y_indices, x_indices] = map[(y_indices + y_add) % len_y, (x_indices + x_add) % len_x]
    # 0の値のインデックスを取得
    y_indices, x_indices = np.where(map == 0)
    # 0の値を置換
    map[y_indices, x_indices] = 1

    # リサイズし1を減算
    map = Image.fromarray(map - 1.0)
    # mapから1を減算してPILイメージを作成
    resized_map = map.resize(resize_shape, Image.LANCZOS)

    return np.asarray(resized_map)


def preprocess_map2(df, normalize_map):
    # データの正規化
    train_maps = np.array([normalize_map(x) for x in df['waferMap']])

    # # 1. 画像を水平方向に反転
    # flipped_horizontally = np.flip(train_maps, axis=2)
    # train_maps = np.concatenate((train_maps, flipped_horizontally), axis=0)

    # # 2. 画像を垂直方向に反転
    # flipped_vertically = np.flip(train_maps, axis=1)
    # train_maps = np.concatenate((train_maps, flipped_vertically), axis=0)

    # 3. 画像を90度回転
    rotated_90 = np.rot90(train_maps, k=1, axes=(1, 2))
    train_maps = np.concatenate((train_maps, rotated_90), axis=0)

    # 4. 画像を180度回転
    rotated_180 = np.rot90(train_maps, k=2, axes=(1, 2))
    train_maps = np.concatenate((train_maps, rotated_180), axis=0)

    # 5. 画像を270度回転
    rotated_270 = np.rot90(train_maps, k=3, axes=(1, 2))
    train_maps = np.concatenate((train_maps, rotated_270), axis=0)

    # データの形状を変更
    train_maps = train_maps.reshape(train_maps.shape + (1,))

    return train_maps


def create_model(input_shape, num_classes):
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        # 畳み込みブロック1
        tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same', input_shape=(input_shape)),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),

        # 畳み込みブロック2
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # 畳み込みブロック3
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # ブロック4
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0),

        # 出力層
        tf.keras.layers.Dense(num_classes),
    ])

    return model

def create_model2(input_shape, num_classes):
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        # 畳み込みブロック1
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(input_shape)),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),

        # 畳み込みブロック2
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # 畳み込みブロック3
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        # ブロック4
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0),

        # 出力層
        tf.keras.layers.Dense(num_classes),
    ])

    return model


def calculate_class_weights(train_labels):
    from sklearn.utils.class_weight import compute_class_weight
    # クラスの重みを計算
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_labels),
        y=train_labels)
    # クラスの重みを辞書型に変換
    return dict(enumerate(class_weights))
    

def solution(x_test_df, train_df):
    import os
    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    failure_types = list(train_df['failureType'].unique())

    test_maps = preprocess_map(x_test_df, normalize_map)
    train_maps = preprocess_map(train_df, normalize_map)
    train_labels = np.array([failure_types.index(x) for x in train_df['failureType']] * 8)

    class_weights = calculate_class_weights(train_labels)

    # モデルの作成
    model = create_model(train_maps[0].shape, len(failure_types))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # EarlyStoppingコールバックの設定
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(train_maps, train_labels, epochs=10, class_weight=class_weights, callbacks=[early_stopping], validation_split=0.2)

    # ニューラルネットワークモデルを使って予測
    test_logits = np.mean(model.predict(test_maps).reshape(-1, len(x_test_df['waferMap']), len(failure_types)), axis=0)
    predictions = tf.nn.softmax(test_logits).numpy()

    answer = [failure_types[x.argmax()] for x in predictions]
    # 低信頼度予測に対する特化したCNNモデルの適用
    low_confidence_indices = [i for i, x in enumerate(predictions) if max(x) < 0.95]
    if low_confidence_indices:
        low_confidence_maps = np.array([test_maps[i] for i in low_confidence_indices])

        # 特化したトレーニングデータの準備（4クラス）
        target_classes = ['Edge-Loc', 'Loc', 'Center', 'Scratch']
        specialized_train_df = train_df[train_df['failureType'].isin(target_classes)]
        specialized_train_maps = preprocess_map2(specialized_train_df, normalize_map2)
        specialized_train_labels = np.array([target_classes.index(ft) for ft in specialized_train_df['failureType']] * 8)

        # 特化したCNNモデルのトレーニング
        specialized_model = create_model2(specialized_train_maps[0].shape, 4)  # 4クラスに対応
        specialized_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        specialized_model.fit(specialized_train_maps, specialized_train_labels, epochs=10, class_weight=class_weights, validation_split=0.2)

        specialized_logits = specialized_model.predict(low_confidence_maps)
        specialized_probabilities = tf.nn.softmax(specialized_logits).numpy()
        specialized_predictions = specialized_probabilities.argmax(axis=1)

        for i, idx in enumerate(low_confidence_indices):
            answer[idx] = target_classes[specialized_predictions[i]]

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
df=pd.read_pickle("../work/input/LSWMD_25519.pkl")

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