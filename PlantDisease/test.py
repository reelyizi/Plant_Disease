import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri seti yolu
dataset_path = 'C:\Users\xbatu\OneDrive\Masaüstü\PlantDisease\plant_diseases_dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)'

# Görüntü boyutlandırma fonksiyonu
def load_and_preprocess_image(img_path, img_size=(128, 128)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalizasyon
    return img

# Tüm görüntüleri ve etiketleri yükleme
def load_dataset(dataset_path):
    images = []
    labels = []
    class_names = os.listdir(dataset_path)
    
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = load_and_preprocess_image(img_path)
            images.append(img)
            labels.append(class_name)
    
    return np.array(images), np.array(labels)

# Veri setini yükle
X, y = load_dataset(dataset_path)

# Veri artırma (augmentation) ayarları
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Eğitim verisine artırma uygulama
datagen.fit(X)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Etiketleri sayısal hale getirme
class_names = np.unique(y_train)
y_train = to_categorical([np.where(class_names == label)[0][0] for label in y_train])
y_test = to_categorical([np.where(class_names == label)[0][0] for label in y_test])

# Model oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=25, validation_data=(X_test, y_test))
