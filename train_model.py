import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set dataset paths (uses your original folder names)
train_dir = os.path.join('dataset', 'train')
val_dir = os.path.join('dataset', 'validation')  # ← updated from 'val'

# Image size and batch size
img_height, img_width = 224, 224
batch_size = 16

# Data generators (with rescaling)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    # Output layer based on number of classes
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# Create model folder if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the trained model
model.save(os.path.join('model', 'model_cnn.h5'))

print("✅ Training complete! Model saved as model/model_cnn.h5")
