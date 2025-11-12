# ReturnBot - AI-Powered Product Return Assistant 🤖

A sophisticated AI-powered chatbot that helps users check product return policies and eligibility using natural language processing and vector database search.


## 🌟 Features

🤖 Smart AI Conversations - Uses pretrained models to understand natural language

📚 Vector Database - Fast and accurate policy retrieval using ChromaDB

🎨 Beautiful UI - Anime-inspired dark theme with smooth animations

💬 Context-Aware - Maintains conversation context and asks relevant follow-up questions

📅 Date Understanding - Automatically extracts and processes purchase dates

✅ Eligibility Check - Determines return eligibility based on policies and conditions

📱 Responsive Design - Works perfectly on desktop and mobile devices

## 🛠️ Technology Stack

### Backend

Python 3.8+ - Core programming language

Flask 2.3.3 - Web framework

LangChain - AI agent framework

ChromaDB - Vector database for policy storage

HuggingFace Transformers - Pretrained language models

Sentence Transformers - Text embeddings

### Frontend

HTML5 & CSS3 - Modern web standards

JavaScript ES6+ - Client-side interactivity

Font Awesome - Icons

CSS Animations - Smooth UI interactions

### AI Models

Microsoft DialoGPT-medium - Primary conversational model

SentenceTransformers/all-MiniLM-L6-v2 - Text embeddings for vector search

## 📦 Available Products in Database

The system currently supports return policies for these products:


| Product	| Return Period	| Key Conditions|
|--------|----------------|---------------|
|💻 Laptop|	30 days	Original packaging | no damage, all accessories|
|📱 Smartphone	|15 days	|Original packaging, no scratches, all accessories|
|🎧 Headphones|	45 days	|Unopened or defective only|
|👕 Clothing	|60 days	|Tags attached, unworn, unwashed|
|📚 Books	|30 days	|No damage, writing, or torn pages|
|🪑 Furniture	|90 days|	Unassembled, no damage or stains|
|📱 Tablet|	30 days	|Original packaging, no damage|
|📷 Camera	|30 days	|Original packaging, shutter count < 1000|

## 🏗️ System Architecture
### Backend Workflow

```
User Input → Flask Route → Smart LLM Manager → Intent Understanding
     ↓
Vector DB Search → Policy Retrieval → Eligibility Check → Response Generation
     ↓
Session Update → JSON Response → Frontend Display

```
## Key Components

SmartLLMManager - Handles AI model loading and natural language understanding

VectorStoreManager - Manages ChromaDB vector database operations

ProductReturnAgent - Core agent that processes conversations and makes decisions

ConversationState - Maintains conversation context across multiple turns

## AI Processing Pipeline
1. User Message → Intent Understanding
   - Extract product type
   - Extract purchase date
   - Extract condition
   - Determine user intent

2. Information Gathering
   - Check missing information
   - Ask follow-up questions if needed

3. Policy Retrieval
   - Search vector database
   - Find relevant policies

4. Eligibility Calculation
   - Compare purchase date with return period
   - Check condition against policy requirements

5. Response Generation
   - Generate natural language response
   - Provide clear eligibility determination
  
## 🚀 Installation & Setup
### Prerequisites

Python 3.8 or higher

4GB+ RAM (for AI models)

2GB+ free disk space

### Set-up:
1. Install Dependencies:
   
```
pip install -r requirements.txt
```

2. Run cmd:
   
```
python app.py
```


### Project Structure:

```
returnbot_app/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── chroma_db/            # Vector database (auto-created)
│
├── templates/
│   └── index.html        # Main HTML template
│
└── static/               # Static assets (CSS, JS, images)
    ├── style.css         # Styles (embedded in HTML)
    └── script.js         # JavaScript (embedded in HTML)

```

### Supported Date Formats
Relative dates: "2 weeks ago", "yesterday", "last month"

Specific dates: "March 15th", "2024-01-15", "15/03/2024"

Simple terms: "today", "just bought it"

### Supported Conditions
"new", "unopened", "sealed"

"used", "opened", "worn"

"good", "perfect", "like new"

"damaged", "broken", "not working"

## 🤖 Model Details

### 📚 Embedding Model:
sentence-transformers/all-MiniLM-L6-v2

Purpose: Converts text (like product policies and user queries) into vector embeddings for semantic similarity search.

Framework: Hugging Face / Sentence Transformers

Dimension: 384

Reason: Lightweight, fast, and widely used for retrieval tasks — perfect for local vector databases.

### 🧩 Vector Database:
Chroma (with FAISS backend)

Purpose: Stores document embeddings and retrieves the most relevant chunks during a query.

Works as the retriever in your RAG pipeline.

### 🗣️ Large Language Model (LLM):
tiiuae/falcon-7b-instruct

Purpose: Generates natural language responses based on the context retrieved from the vector store.

Type: Instruction-tuned causal language model (7 billion parameters).

Hosted Locally (not API-dependent).

Known for: Strong performance on reasoning and question-answering tasks.

### Policy Documents Structure
Each policy document contains:
```
Product: [Product Name]
Return Period: [Timeframe]
Conditions: [Requirements]
Special Notes: [Additional information]
```

### Data Storage Location
Vector database: ./chroma_db/

Session data: Flask server memory (resets on server restart)

### ⚡ Performance
Response Time: 2-5 seconds (including AI processing)

Vector Search: <100ms

Memory Usage: ~1.5GB (with models loaded)

Concurrent Users: Limited by Flask development server

### 🔒 Security Features
Session-based conversation management

Input sanitization and validation

No sensitive data storage

Local processing (no external API calls required)

## 🐛 Troubleshooting

### Common Issues

* Model Download Failures : Solution - Check internet connection, retry installation
* Memory Errors :Solution - Close other applications, increase system RAM


## 📈 Future Enhancements
User authentication and history persistence

Multi-language support

Integration with real product databases

Advanced AI models (GPT-3.5/4 with API)

Email notifications for return status

Admin dashboard for policy management

Voice interface support


## 🙏 Acknowledgments
Microsoft for DialoGPT model

HuggingFace for transformer models and ecosystem

ChromaDB for vector database

LangChain for AI agent framework

Flask for web framework

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


~ Sahil Shaikh (SS-2005) All Rights Reserved



