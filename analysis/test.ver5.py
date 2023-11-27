import numpy as np # https://numpy.org/ja/
import pandas as pd # https://pandas.pydata.org/
from sklearn.model_selection import train_test_split


def normalize_map(map):
    from PIL import Image

    # リサイズ後のサイズを指定
    resize_shape = (28, 28)
    
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


def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add

    # Shortcut path
    shortcut = x
    if conv_shortcut:  # If the shortcut needs to be resized
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = BatchNormalization(name=name + '_0_bn')(shortcut)

    # Main path
    x = Conv2D(filters, kernel_size, padding='same', strides=stride, name=name + '_1_conv')(x)
    x = BatchNormalization(name=name + '_1_bn')(x)
    x = ReLU(name=name + '_1_relu')(x)

    x = Conv2D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)
    x = BatchNormalization(name=name + '_2_bn')(x)
    x = ReLU(name=name + '_2_relu')(x)

    x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = BatchNormalization(name=name + '_3_bn')(x)

    # Merge paths
    x = Add(name=name + '_add')([shortcut, x])
    x = ReLU(name=name + '_out')(x)
    return x


# 新しいモデル作成関数
def create_residual_model(input_shape, num_classes):
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Add residual blocks
    x = residual_block(x, 64, stride=2, conv_shortcut=True, name='res_block1')
    # Add more blocks as needed...

    # Global Average Pooling and output layer
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs, outputs)
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
    failure_types = list(train_df['failureType'].unique())
    test_maps = preprocess_map(x_test_df, normalize_map)
    train_maps = preprocess_map(train_df, normalize_map)
    train_labels = np.array([failure_types.index(x) for x in train_df['failureType']] * 8)
    class_weights = calculate_class_weights(train_labels)

    # 新しい残差モデルを作成
    model = create_residual_model(train_maps[0].shape, len(failure_types))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # EarlyStoppingコールバックの設定
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    # モデルのトレーニング
    model.fit(train_maps, train_labels, epochs=7, class_weight=class_weights, callbacks=[early_stopping], validation_split=0.2)

    # 各予測結果の平均を計算
    test_logits = np.mean(model.predict(test_maps).reshape(-1, len(x_test_df['waferMap']), len(failure_types)), axis=0)
    
    # 予測値を取得
    predictions = tf.nn.softmax(test_logits).numpy()
    answer = [failure_types[x.argmax()] for x in predictions]
    
    # 予測結果をデータフレームで返す
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