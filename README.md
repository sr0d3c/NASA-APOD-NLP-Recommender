# 🌌 StellarMatch - NASA APOD Content-Based Recommender

A sophisticated content-based recommendation system for NASA's Astronomy Picture of the Day (APOD) dataset, powered by Natural Language Processing and machine learning. Discover astronomical wonders similar to your favorites through semantic analysis of image descriptions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Application Interface](#application-interface)
- [Technical Stack](#technical-stack)
- [Data Pipeline](#data-pipeline)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

**StellarMatch** is an intelligent recommendation engine that analyzes NASA's APOD dataset to find visually and thematically similar astronomical images. By leveraging:

- **TF-IDF Vectorization** for semantic text analysis
- **Cosine Similarity** for content matching
- **spaCy NLP** for advanced text processing
- **Streamlit** for interactive visualization

Users can explore thousands of astronomical images and discover related content based on semantic similarity rather than traditional keyword matching.

---

## Features

✨ **Key Capabilities:**

- 🔍 **Semantic Search**: Find similar astronomical images based on content descriptions
- 📊 **NLP Analytics**: Real-time visualization of dominant themes in the dataset and recommendations
- 🖼️ **Image Gallery**: Beautiful card-based interface displaying images and metadata
- 📈 **Theme Analysis**: Keyword frequency charts showing global and recommendation-specific themes
- 🚀 **Fast Inference**: Pre-computed cosine similarity matrix for instant recommendations
- 📱 **Responsive Design**: Wide layout optimized for desktop exploration

---

## Project Structure

```
NASA-APOD-NLP-Recommender/
├── app.py                          # Main Streamlit application
├── data_processing.ipynb           # Data enrichment and model training pipeline
├── apod_enriched_data.csv          # Processed dataset with explanations
├── stellarmatch_model.pkl          # Pre-trained model components
├── requirements.txt                # Python dependencies
├── apod_data/
│   ├── infos.csv                   # Original APOD metadata
│   └── APOC/                       # Image files (JPG format)
└── README.md                       # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- NASA API key (for updating data)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/NASA-APOD-NLP-Recommender.git
cd NASA-APOD-NLP-Recommender
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n stellarmatch python=3.10
conda activate stellarmatch
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm
```

### Step 4: Configure NASA API (Optional)

Create a `.env` file in the project root:

```
NASA_API_KEY=your_api_key_here
APOD_BASE_URL=https://api.nasa.gov/planetary/apod
```

Get your API key at: [NASA Open APIs](https://api.nasa.gov/)

---

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Basic Workflow

1. **Select an Image**: Choose an astronomical object from the dropdown menu
2. **Generate Recommendations**: Click the "Generate Recommendations" button
3. **Explore Results**: View your selection and 5 similar astronomical images
4. **Analyze Themes**: Check the sidebar for keyword analysis of recommendations

---

## How It Works

### Pipeline Overview

```
Raw APOD Data
    ↓
Data Enrichment (NASA API)
    ↓
Text Preprocessing (spaCy)
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity Computation
    ↓
Model Serialization (.pkl)
    ↓
Streamlit Recommendation Interface
```

### Core Algorithm

1. **Data Enrichment**: Fetch detailed explanations from NASA API for each image
2. **Text Processing**: 
   - Tokenization and POS tagging with spaCy
   - Removal of stopwords and special characters
   - Lemmatization for semantic consistency

3. **Feature Extraction**: Convert cleaned text to TF-IDF vectors
4. **Similarity Matching**: Compute cosine similarity between all image pairs
5. **Recommendation**: Return top-N most similar images based on similarity scores

---

## Application Interface

### Home View

When you first launch the application, you'll see:

```
┌─────────────────────────────────────────────────────────────┐
│  🌌 StellarMatch                                            │
│  ### Content-Based Discovery for NASA's APOD Dataset        │
│                                                              │
│  Search for an astronomical object: [Dropdown Menu ▼]      │
│  [Generate Recommendations] (Primary Button)               │
│                                                              │
│  💡 Select an image above to get started!                  │
└─────────────────────────────────────────────────────────────┘

Sidebar:
┌─────────────────────────────┐
│ [NASA Logo]                 │
│ Project Insights            │
│                             │
│ Library size: 700+ images │
│                             │
│ Global Catalog Themes       │
│ [Bar Chart of Keywords]     │
│                             │
│ Most frequent terms across  │
│ the entire NASA dataset.    │
└─────────────────────────────┘
```

### Results View (After Selection)

Once you select an image and generate recommendations:

```
┌─────────────────────────────────────────────────────────────┐
│                      Your selection                         │
│  ┌────────────────┐                                         │
│  │  [Image 1]     │                                         │
│  │  Title         │                                         │
│  │  Date: YYYY-MM-DD                                        │
│  └────────────────┘                                         │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Similar wonders you might like                             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │
│  │ Image2 │ │ Image3 │ │ Image4 │ │ Image5 │ │ Image6 │  │
│  │ Title  │ │ Title  │ │ Title  │ │ Title  │ │ Title  │  │
│  │ 📋View │ │ 📋View │ │ 📋View │ │ 📋View │ │ 📋View │  │
│  │Details │ │Details │ │Details │ │Details │ │Details │  │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │
└─────────────────────────────────────────────────────────────┘

Sidebar (Dynamic):
┌─────────────────────────────┐
│ Project Insights            │
│ Library size: 700+ images │
│                             │
│ Global Catalog Themes       │
│ [Bar Chart]                 │
│                             │
│ ─────────────────────────── │
│ Current Match Keywords      │
│ [Red Bar Chart]             │
│                             │
│ Dominant terms within your  │
│ 5 recommendations           │
│ (Semantic Context)          │
└─────────────────────────────┘
```

### Interactive Elements

- **Dropdown Selection**: Searchable list of all 700+ astronomical objects
- **Generate Button**: Trigger recommendation computation
- **Expandable Details**: Click "View Details" on each recommendation card to see:
  - Observation date (extracted from filename)
  - Full explanation text
  - NASA copyright information

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit 1.28+ | Interactive web interface |
| **NLP** | spaCy 3.8.0 | Advanced text processing |
| **ML** | scikit-learn | TF-IDF & Cosine Similarity |
| **Data** | pandas, NumPy | Data manipulation |
| **Caching** | Streamlit cache_resource | Fast model loading |
| **Image Processing** | Pillow | Image display |
| **API** | requests | NASA API integration |

---

## Data Pipeline

### Input Data

**apod_data/infos.csv**
- Source: [NASA APOD Dataset on Kaggle](https://www.kaggle.com/datasets/melcore/astronomy-picture-of-the-day/data)
- Format: CSV with Title and Filename columns
- Original Data: NASA Astronomy Picture of the Day (APOD)

### Processing Steps (data_processing.ipynb)

1. **Data Loading**: Read CSV and remove duplicates
2. **API Enrichment**: Fetch explanations from NASA APOD API
3. **Text Cleaning**:
   - Lowercase conversion
   - Special character removal
   - Stopword filtering
   - Lemmatization

4. **Vectorization**: Apply TF-IDF vectorizer
5. **Similarity Computation**: Build cosine similarity matrix
6. **Model Export**: Serialize components to `stellarmatch_model.pkl`

### Output Format

The pickle file contains:
```python
{
    'dataframe': pd.DataFrame,        # Full dataset with explanations
    'cosine_sim': np.ndarray,         # Precomputed similarity matrix
    'indices': pd.Series              # Title to index mapping
}
```

---

## Running the Data Pipeline

To update the model with new data:

```bash
jupyter notebook data_processing.ipynb
```

**Important**: Set `SAVE_EVERY` to control checkpoint frequency when processing the API (recommended: 50)

### Progress Tracking

- Automatically resumes from last checkpoint if interrupted
- Saves progress every N records to `apod_enriched_data.csv`
- Includes rate limiting for NASA API compliance

---

## File Descriptions

### app.py
Main Streamlit application containing:
- Page configuration and caching
- NLP analytics functions (keyword extraction)
- Recommendation algorithm
- UI components (sidebar, image cards)
- Main interface orchestration

### data_processing.ipynb
Jupyter notebook containing:
- Data loading and preprocessing
- NASA API integration
- spaCy text processing
- TF-IDF vectorization
- Model training and serialization
- Progress tracking and resumption logic

### apod_enriched_data.csv
Enriched dataset with:
- Original APOD metadata
- Full explanation texts from NASA API
- Cleaned text for NLP processing
- Used as source for model training

---

## Troubleshooting

### Common Issues

**Issue**: "Model file not found" error
```
Solution: Ensure stellarmatch_model.pkl exists in the project root.
Run data_processing.ipynb to regenerate the model.
```

**Issue**: Images not displaying
```
Solution: Verify APOC folder exists with .jpg files.
Check file naming matches Filename column in CSV.
```

**Issue**: Slow recommendations
```
Solution: Clear Streamlit cache: rm -rf ~/.streamlit/cache*
Restart the application.
```

**Issue**: NASA API rate limiting
```
Solution: Increase delays in data_processing.ipynb
Implement caching in the API calls
```

---

## License

This project is licensed under the MIT License - see the License file for details.

### Attribution

- Data Source: [NASA Astronomy Picture of the Day](https://apod.nasa.gov/)
- Images: Copyright NASA/APOD (Public Domain)
- Icons: Streamlit built-in emoji support

---

## Acknowledgments

- 📊 Dataset Source: [Kaggle - Astronomy Picture of the Day](https://www.kaggle.com/datasets/melcore/astronomy-picture-of-the-day/data)
- NASA for providing the APOD API and original dataset

---

**Last Updated**: February 2026  
**Version**: 1.1.0  