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

    # 1. 画像を水平方向に反転
    flipped_horizontally = np.flip(train_maps, axis=2)
    train_maps = np.concatenate((train_maps, flipped_horizontally), axis=0)

    # # 2. 画像を垂直方向に反転
    # flipped_vertically = np.flip(train_maps, axis=1)
    # train_maps = np.concatenate((train_maps, flipped_vertically), axis=0)

    # 3. 画像を90度回転
    rotated_90 = np.rot90(train_maps, k=1, axes=(1, 2))
    train_maps = np.concatenate((train_maps, rotated_90), axis=0)

    # 4. 画像を180度回転
    rotated_180 = np.rot90(train_maps, k=2, axes=(1, 2))
    train_maps = np.concatenate((train_maps, rotated_180), axis=0)

    # # 5. 画像を270度回転
    # rotated_270 = np.rot90(train_maps, k=3, axes=(1, 2))
    # train_maps = np.concatenate((train_maps, rotated_270), axis=0)

    # データの形状を変更
    train_maps = train_maps.reshape(train_maps.shape + (1,))

    return train_maps


def preprocess_size(df):
    import tensorflow as tf

    # dieSizeの正規化（0-1の間にスケーリング）
    size = df['dieSize'].values
    size = size / tf.reduce_max(size)

    return size


def create_hybrid_model(map_shape, num_classes):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate, Multiply
    from tensorflow.keras.models import Model

    # waferMap用の入力レイヤー
    wafermap_input = Input(shape=map_shape, name='wafermap_input')
    
    # 畳み込みとプーリング層
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(wafermap_input)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(40, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    wafermap_features = Flatten()(x)

    # dieSize用の入力レイヤー
    diesize_input = Input(shape=(1,), name='diesize_input')
    diesize_dense = Dense(16, activation='relu')(diesize_input)

    # 注意機構の適用
    # `wafermap_features`の長さに合わせて`attention_probs`を変換
    attention_dense = Dense(tf.keras.backend.int_shape(wafermap_features)[-1], activation='softmax', name='attention_vec')(diesize_dense)
    attention_mul = Multiply()([wafermap_features, attention_dense])

    # 特徴量の結合
    combined_features = concatenate([attention_mul, diesize_dense])

    # 密結合層とドロップアウト
    x = Dense(320, activation='relu')(combined_features)
    x = Dropout(0.1)(x)
    x = Dense(80, activation='relu')(x)
    x = Dropout(0.1)(x)

    # 出力層
    outputs = Dense(num_classes)(x)

    # モデルの定義
    model = Model(inputs=[wafermap_input, diesize_input], outputs=outputs)
    
    return model


def calculate_class_weights(train_labels):
    from sklearn.utils.class_weight import compute_class_weight
    # クラスの重みを計算
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(train_labels), 
                                         y=train_labels)
    # class_weights[4] *= 1.5
    # クラスの重みを辞書型に変換
    return dict(enumerate(class_weights))


def solution(x_test_df, train_df):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import tensorflow as tf
    #import keras_tuner as kt

    # failureTypeのユニークな値を取得
    failure_types = list(train_df['failureType'].unique())

    test_maps = preprocess_map(x_test_df, normalize_map)
    train_maps = preprocess_map(train_df, normalize_map)
    test_sizes = np.repeat(preprocess_size(x_test_df), 8)
    train_sizes = np.repeat(preprocess_size(train_df), 8)
    train_labels = np.array([failure_types.index(x) for x in train_df['failureType']] * 8)

    class_weights = calculate_class_weights(train_labels)

    model = create_hybrid_model(train_maps[0].shape, len(failure_types))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit([train_maps, train_sizes], train_labels, epochs=5, class_weight=class_weights)

    # tuner = kt.RandomSearch(
    #     create_model,
    #     objective='val_accuracy',
    #     max_trials=10,
    #     directory='tuner',
    #     project_name='wafermap'
    # )

    # tuner.search(train_maps, train_labels, epochs=10, validation_split=0.1)

    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # model = tuner.hypermodel.build(best_hps)
    # model.fit(train_maps, train_labels, epochs=10, class_weight=class_weights)

    # 各予測結果の平均を計算
    test_logits = np.mean(model.predict([test_maps, test_sizes]).reshape(-1, len(x_test_df['waferMap']), len(failure_types)), axis=0)
    predictions = tf.nn.softmax(test_logits).numpy()
    answer = [failure_types[x.argmax()] for x in predictions]

    return pd.DataFrame({'failureType': answer}, index=x_test_df.index)