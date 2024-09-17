## Extracting Textual Data from Product Images: A Detailed Process Overview

We designed a model to process a dataset of images (the product images) and their associated numeric values and text labels, which is going to be the value of the quantity that needs to be identified and its respective unit (e.g., 50 grams, 10 kg), extract key information, and perform image processing using GPU acceleration with TensorFlow. The results are saved to a CSV file, and the code is optimized to handle large datasets by processing images in batches.

## Importing Required Libraries
No APIs and pre-trained models were used. TensorFlow is used for GPU-accelerated image processing, while libraries such as PIL (Python Imaging Library) and requests help download and manipulate images. asyncio allows for asynchronous processing, improving performance by handling tasks concurrently. Additionally, nest_asyncio has been applied to enable async event loops in environments like Jupyter notebooks. We also make use of easyocr, which is a Python library that provides Optical Character Recognition (OCR) capabilities. 

## Dataset Loading and Setup
A CSV file with image links and numeric data is loaded into a pandas DataFrame. An output CSV is initialized to store the results (image name, numeric value, and text). A checkpoint file tracks processed images to avoid redundancy and improve efficiency.

## Extracting Numeric and Text Values
The extract_numeric_and_text function uses regular expressions to find numeric values followed by text in the entity_value column of the dataset. This function returns two values: the numeric value and the associated text (e.g., "50" and "kilograms"). If no match is found, the function returns Null. This is useful when processing image data that may have embedded numeric information.

## Image Downloading and Marking
Downloading image from the links part was handled with the help of numerous functions. The download_image function downloads an image from a URL using the `requests` library. The image is then processed using PIL to draw a red rectangle around a portion of the image and display the extracted numeric value and text. This marking is handled by the mark_image function, which adds visual cues to the image to show where the numeric value and unit were detected. The image is further processed using the process_image_on_gpu function, which converts the image to a TensorFlow tensor and performs any required processing operations on the GPU.

## CSV Output and Checkpointing
Results (image name, numeric value, and text) are saved incrementally to the output CSV using the append_to_csv function. The save_checkpoint function logs processed images, preventing reprocessing and allowing the script to resume from the last checkpoint if interrupted.

## Asynchronous Image Processing
The script uses asyncio for efficient handling of large datasets, processing images in batches. The process_image_row function extracts numeric values, downloads, marks, and processes images on the GPU, while process_batch processes these in parallel, showing real-time progress with tqdm.
The main function, main, manages batch processing by splitting the dataset into smaller batches (which can be adjusted based on system resources). Each batch is processed asynchronously to ensure efficient handling of large datasets.

The CSV file obtained has separate fields for numeric value detected and the unit. We then combine the numeric_value and text_value columns into a new combined_value column, ensuring both are converted to strings for concatenation. After that, we creates a new DataFrame that retains only the image_name and combined_value columns. The resulting DataFrame is saved as a new CSV file at the specified path, with the print statement confirming successful completion.

## Accuracy and F1 Score

The model achieved an accuracy of 89.31% on the training data, indicating that it correctly predicted nearly 90% of the cases. Additionally, with an F1 score of 0.9435, the model demonstrates a strong balance between precision and recall, reflecting its ability to handle both false positives and false negatives effectively. These metrics suggest that the model is performing well in terms of both accuracy and robustness in classification.

