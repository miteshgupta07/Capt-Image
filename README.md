<div align='center'>
  <h1>🖼️✍️ CAPT-IMAGE 🖼️✍️</h1>
</div>

**CAPT-IMAGE** is an advanced image captioning tool that generates descriptive captions for images using deep learning techniques. It utilizes a combination of VGG16 for feature extraction and LSTM for generating meaningful captions.

## Features 🌟
- **Image Upload**: Supports various formats including JPG, PNG, WEBP, and SVG.
- **Caption Generation**: Provides descriptive captions based on the content of the uploaded image.
- **User-Friendly Interface**: Built with Streamlit for a seamless and interactive user experience.

## Installation 🛠️

1. **Clone the repository:**
   ```bash
   git clone https://github.com/miteshgupta07/Capt-Image.git

2. **Navigate to the project directory:**
   ```bash
   cd Capt-Image

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt

## Usage 🚀
1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py

2. **Upload an image and view the generated caption.**

## Files 📁

• **app.py:** Main Streamlit application file.

• **vgg_model.keras:** Pre-trained VGG16 model used for feature extraction.

• **model.keras:** LSTM model used for caption generation.

• **Tokenizer.pkl:** Tokenizer object for text processing.

• **Extracted_Feature.pkl:** Extracted Feature File.

## Dataset 📚
The dataset used for training the models is the **Flickr8k** dataset, which includes a collection of images with corresponding captions. The dataset is publicly available and can be accessed [here](https://www.kaggle.com/datasets/adityajn105/flickr8k).

## Model Details 🔍

• **Feature Extraction Model:** VGG16

• **Caption Generation Model:** LSTM with dense layers

## Contribution 🤝
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

## License 📝
This project is licensed under the MIT License - see the [LICENSE](https://github.com/miteshgupta07/Capt-Image/blob/main/LICENSE) file for details.

## Contact 📧
For any questions or feedback, feel free to reach out at miteshgupta2711@gmail.com.
