import os
import cv2

# Input and Output paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# Target image size
IMG_SIZE = 256


def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize to [0,1]
    img = img / 255.0

    return img


def process_dataset():
    for split in ["train", "test"]:
        split_path = os.path.join(RAW_DATA_PATH, split)

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)

            # Create corresponding output folder
            output_class_path = os.path.join(PROCESSED_DATA_PATH, split, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                try:
                    processed_img = preprocess_image(img_path)

                    # Convert back to 0–255 for saving
                    processed_img = (processed_img * 255).astype('uint8')

                    # Save as PNG
                    save_path = os.path.join(
                        output_class_path,
                        os.path.splitext(img_name)[0] + ".png"
                    )

                    cv2.imwrite(save_path, processed_img)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    process_dataset()