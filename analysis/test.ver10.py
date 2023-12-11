import numpy as np # https://numpy.org/ja/
import pandas as pd # https://pandas.pydata.org/
from sklearn.model_selection import train_test_split

# !pip install numpy
# !pip install pandas
# !pip install scikit-learn
# !pip install tensorflow
# !pip install Pillow

def solution(x_test_df, train_df):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from PIL import Image
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Add
    from tensorflow.keras.models import Model

    def resize_map(map):

        resize_shape = (32, 32)

        non_zero_average = np.mean(map[map != 0])
        map[map == 0] = non_zero_average
    
        map = Image.fromarray(map - 1.0)
        resized_map = map.resize(resize_shape, Image.LANCZOS)

        return np.asarray(resized_map)

    
    def preprocess_map(df, resize_map):

        preprocessed_maps = np.array([resize_map(x) for x in df['waferMap']])

        rotated_90 = np.rot90(preprocessed_maps, k=1, axes=(1, 2))
        preprocessed_maps = np.concatenate((preprocessed_maps, rotated_90), axis=0)

        rotated_180 = np.rot90(preprocessed_maps, k=2, axes=(1, 2))
        preprocessed_maps = np.concatenate((preprocessed_maps, rotated_180), axis=0)

        flipped_horizontally = np.flip(preprocessed_maps, axis=2)
        preprocessed_maps = np.concatenate((preprocessed_maps, flipped_horizontally), axis=0)

        preprocessed_maps = preprocessed_maps.reshape(preprocessed_maps.shape + (1,))

        return preprocessed_maps

    
    def initialize_cnn(input_shape, failure_types_classes):
        inputs = Input(shape=input_shape)
    
        x = Conv2D(16, 3, activation='relu', padding='same')(inputs)
        x = MaxPooling2D(pool_size=2, strides=2)(x)
        skip_connection = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=2, strides=2)(x)
        x = Add()([x, skip_connection])

        x = Conv2D(128, 1, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=2, strides=2)(x)
    
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
    
        outputs = Dense(failure_types_classes)(x)
    
        model = Model(inputs=inputs, outputs=outputs)
    
        return model


    def calculate_class_weights(train_labels):

        class_weights = compute_class_weight(class_weight='balanced', 
                                             classes=np.unique(train_labels), 
                                             y=train_labels)

        return dict(enumerate(class_weights))


    failure_types = list(train_df['failureType'].unique())

    train_maps = preprocess_map(train_df, resize_map)
    train_labels = np.array([failure_types.index(x) for x in train_df['failureType']] * 8)

    failure_types_classes = len(failure_types)
    input_shape = train_maps[0].shape

    class_weights = calculate_class_weights(train_labels)

    model = initialize_cnn(input_shape, failure_types_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00020),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(train_maps, train_labels, epochs=10, batch_size=64, class_weight=class_weights)

    test_maps = preprocess_map(x_test_df, resize_map)
    map_classes = len(x_test_df['waferMap'])
    
    test_predictions = model.predict(test_maps)
    aggregated_logits = np.zeros(map_classes * failure_types_classes, dtype=np.float64).reshape((map_classes, failure_types_classes))
    for n in range(8):
        aggregated_logits += test_predictions[map_classes * n  :map_classes * (n + 1)]
    
    predictions = tf.nn.softmax(aggregated_logits).numpy()
    answer = [failure_types[x.argmax()] for x in predictions]
    
    return pd.DataFrame({'failureType': answer}, index=x_test_df.index)

# データのインポート
df = pd.read_pickle("../work/input/LSWMD_25519.pkl")

# テスト用と学習用のデータを作成
train_df, test_df = train_test_split(df, stratify=df['failureType'], test_size=0.10, random_state=42)

y_test_df = test_df[['failureType']]
x_test_df = test_df.drop(columns=['failureType'])

# 10回の繰り返しでsolution関数を実行し、平均精度を計算
average_accuracies = []
for i in range(10):
    # solution関数を実行
    user_result_df = solution(x_test_df, train_df)

    # 平均精度の計算
    if type(y_test_df) == type(user_result_df) and y_test_df.shape == user_result_df.shape:
        accuracies = {}
        for failure_type in df['failureType'].unique():
            y_test_df_by_failure_type = y_test_df[y_test_df['failureType'] == failure_type]
            user_result_df_by_failure_type = user_result_df[y_test_df['failureType'] == failure_type]
            matching_rows = (y_test_df_by_failure_type == user_result_df_by_failure_type).all(axis=1).sum()
            accuracies[failure_type] = (matching_rows / len(y_test_df_by_failure_type))

        average_accuracy = sum(accuracies.values()) / len(accuracies)
        average_accuracies.append(average_accuracy)

# 10回の繰り返しの平均精度を計算
final_average_accuracy = sum(average_accuracies) / len(average_accuracies)
print(f"10回の平均精度：{final_average_accuracy * 100:.2f}%")
