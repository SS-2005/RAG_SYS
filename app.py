from flask import Flask, render_template, request, jsonify, session
from datetime import datetime, timedelta
import json
import os
import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain and Vector DB imports
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_change_in_production_2024'
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Initialize components
class SmartLLMManager:
    def __init__(self):
        logger.info("🔄 Loading smart conversational model...")
        try:
            # Using a model that's good for conversations and instruction following
            model_name = "microsoft/DialoGPT-medium"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create text generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=200,
                temperature= 0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            self.model_loaded = True
            logger.info("✅ Smart LLM loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model_loaded = False
    
    def understand_user_intent(self, user_message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Use the model to understand user intent and extract information"""
        if not self.model_loaded:
            return self._rule_based_understanding(user_message, conversation_history)
        
        try:
            # Create context from conversation history
            context = self._build_conversation_context(conversation_history)
            
            prompt = f"""Analyze this customer service conversation about product returns:

{context}
User: {user_message}

Extract the following information:
1. Product mentioned (laptop, smartphone, headphones, clothing, books, furniture, tablet, camera)
2. Purchase date or time mentioned
3. Product condition mentioned
4. User's main intent (check_return_policy, check_eligibility, general_question)

Respond in JSON format:
{{
    "product": "product_name_or_empty",
    "purchase_date": "date_or_empty", 
    "condition": "condition_or_empty",
    "intent": "user_intent",
    "needs_followup": true_or_false,
    "missing_info": ["what's_missing"]
}}"""

            response = self.pipe(prompt, max_length=300)[0]['generated_text']
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(f"📊 Model understanding: {result}")
                return result
            else:
                return self._rule_based_understanding(user_message, conversation_history)
                
        except Exception as e:
            logger.error(f"Model understanding error: {e}")
            return self._rule_based_understanding(user_message, conversation_history)
    
    def generate_conversational_response(self, prompt: str, context: str = "") -> str:
        """Generate natural conversational response"""
        if not self.model_loaded:
            return self._rule_based_response(prompt, context)
        
        try:
            full_prompt = f"Context: {context}\n\nPrompt: {prompt}\nResponse:"
            response = self.pipe(full_prompt, max_length=250)[0]['generated_text']
            # Extract only the response part
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._rule_based_response(prompt, context)
    
    def _build_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Build context from conversation history"""
        if not conversation_history:
            return "New conversation"
        
        context_lines = []
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            sender = "User" if msg['sender'] == 'user' else "Assistant"
            context_lines.append(f"{sender}: {msg['message']}")
        
        return "\n".join(context_lines)
    
    def _rule_based_understanding(self, user_message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Fallback rule-based understanding"""
        user_message_lower = user_message.lower()
        
        # Extract product
        product = ""
        products = {
            "laptop": ["laptop", "computer", "macbook", "notebook"],
            "smartphone": ["smartphone", "phone", "iphone", "android", "mobile"],
            "headphones": ["headphone", "earphone", "earbud", "headset"],
            "clothing": ["clothing", "shirt", "pants", "dress", "jacket", "jeans"],
            "books": ["book", "novel", "textbook", "ebook"],
            "furniture": ["furniture", "chair", "table", "sofa", "couch", "desk"],
            "tablet": ["tablet", "ipad", "android tablet"],
            "camera": ["camera", "dslr", "mirrorless"]
        }
        
        for prod, keywords in products.items():
            if any(keyword in user_message_lower for keyword in keywords):
                product = prod
                break
        
        # Extract purchase date
        purchase_date = self._extract_date_from_text(user_message)
        
        # Extract condition
        condition = ""
        if any(word in user_message_lower for word in ["broken", "damaged", "cracked", "not working", "defective"]):
            condition = "damaged"
        elif any(word in user_message_lower for word in ["good", "perfect", "like new", "excellent", "mint"]):
            condition = "good"
        elif any(word in user_message_lower for word in ["used", "opened", "worn", "normal wear"]):
            condition = "used"
        elif any(word in user_message_lower for word in ["new", "unopened", "sealed"]):
            condition = "new"
        
        # Determine intent
        intent = "general_question"
        if any(word in user_message_lower for word in ["return", "send back", "exchange", "refund"]):
            if purchase_date or condition:
                intent = "check_eligibility"
            else:
                intent = "check_return_policy"
        
        # Determine missing info
        missing_info = []
        if intent in ["check_eligibility", "check_return_policy"]:
            if not product:
                missing_info.append("product")
            if intent == "check_eligibility" and not purchase_date:
                missing_info.append("purchase_date")
            if intent == "check_eligibility" and not condition:
                missing_info.append("condition")
        
        return {
            "product": product,
            "purchase_date": purchase_date,
            "condition": condition,
            "intent": intent,
            "needs_followup": len(missing_info) > 0,
            "missing_info": missing_info
        }
    
    def _extract_date_from_text(self, text: str) -> str:
        """Extract date information from text"""
        text_lower = text.lower()
        now = datetime.now()
        
        # Relative dates
        if "today" in text_lower or "just" in text_lower:
            return now.strftime("%Y-%m-%d")
        elif "yesterday" in text_lower:
            return (now - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "week" in text_lower:
            if "last" in text_lower or "1" in text_lower:
                return (now - timedelta(weeks=1)).strftime("%Y-%m-%d")
            elif "2" in text_lower:
                return (now - timedelta(weeks=2)).strftime("%Y-%m-%d")
            elif "3" in text_lower:
                return (now - timedelta(weeks=3)).strftime("%Y-%m-%d")
        elif "month" in text_lower:
            if "1" in text_lower or "one" in text_lower:
                return (now - timedelta(days=30)).strftime("%Y-%m-%d")
            elif "2" in text_lower:
                return (now - timedelta(days=60)).strftime("%Y-%m-%d")
        
        # Specific date patterns
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
            r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{2,4})',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})\s+(\d{2,4})'
        ]
        
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
            'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                if pattern == date_patterns[0]:  # MM/DD/YYYY
                    month, day, year = match.groups()
                    if len(year) == 2:
                        year = f"20{year}"
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                elif pattern == date_patterns[1]:  # DD Month YYYY
                    day, month, year = match.groups()
                    month_num = month_map[month.lower()[:3]]
                    if len(year) == 2:
                        year = f"20{year}"
                    return f"{year}-{month_num}-{day.zfill(2)}"
                elif pattern == date_patterns[2]:  # Month DD YYYY
                    month, day, year = match.groups()
                    month_num = month_map[month.lower()[:3]]
                    if len(year) == 2:
                        year = f"20{year}"
                    return f"{year}-{month_num}-{day.zfill(2)}"
        
        return ""
    
    def _rule_based_response(self, prompt: str, context: str) -> str:
        """Fallback rule-based response"""
        return "I understand you're asking about return policies. Let me help you with that!"

# Vector Store Manager
class VectorStoreManager:
    def __init__(self, collection_name="product_policies"):
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def initialize_vector_store(self, documents):
        """Initialize ChromaDB with product documents"""
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db",
            collection_name=self.collection_name
        )
        return vector_store
    
    def get_retriever(self, k=3):
        """Get retriever for similarity search"""
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        return vector_store.as_retriever(search_kwargs={"k": k})
    
    def get_all_products(self) -> List[str]:
        """Get list of all products in the database"""
        try:
            vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            # Get all documents
            all_docs = vector_store.get()
            products = set()
            for doc in all_docs['documents']:
                # Extract product name from document
                if "Product:" in doc:
                    product_line = [line for line in doc.split('\n') if "Product:" in line][0]
                    product_name = product_line.split("Product:")[1].strip()
                    products.add(product_name)
            return sorted(list(products))
        except Exception as e:
            logger.error(f"Error getting products: {e}")
            return ["laptop", "smartphone", "headphones", "clothing", "books", "furniture", "tablet", "camera"]

# Conversation State
@dataclass
class ConversationState:
    current_product: str = ""
    purchase_date: str = ""
    product_condition: str = ""
    current_intent: str = ""
    missing_info: List[str] = None
    conversation_history: List[Dict] = None
    
    def __post_init__(self):
        if self.missing_info is None:
            self.missing_info = []
        if self.conversation_history is None:
            self.conversation_history = []
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    
    def reset(self):
        """Reset conversation state"""
        self.current_product = ""
        self.purchase_date = ""
        self.product_condition = ""
        self.current_intent = ""
        self.missing_info = []
        # Keep conversation history for context

# Smart Product Return Agent
class SmartProductReturnAgent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm_manager = SmartLLMManager()
        self.vector_manager = VectorStoreManager()
    
    def process_message(self, user_message: str, conversation_state: ConversationState) -> Dict[str, Any]:
        """Process user message with smart understanding"""
        logger.info(f"Processing: {user_message}")
        
        # Understand user intent and extract information
        understanding = self.llm_manager.understand_user_intent(
            user_message, 
            conversation_state.conversation_history
        )
        
        # Update conversation state with extracted information
        self._update_conversation_state(understanding, conversation_state)
        
        # Add user message to history
        conversation_state.conversation_history.append({
            'sender': 'user', 
            'message': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate appropriate response
        if understanding['needs_followup']:
            response = self._generate_followup_question(understanding, conversation_state)
            is_followup = True
        else:
            response = self._generate_final_response(understanding, conversation_state)
            is_followup = False
        
        # Add agent response to history
        conversation_state.conversation_history.append({
            'sender': 'agent', 
            'message': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'response': response,
            'is_followup': is_followup,
            'conversation_state': conversation_state.to_dict(),
            'understanding': understanding
        }
    
    def _update_conversation_state(self, understanding: Dict, state: ConversationState):
        """Update conversation state with new information"""
        if understanding['product']:
            state.current_product = understanding['product']
        if understanding['purchase_date']:
            state.purchase_date = understanding['purchase_date']
        if understanding['condition']:
            state.product_condition = understanding['condition']
        
        state.current_intent = understanding['intent']
        state.missing_info = understanding['missing_info']
    
    def _generate_followup_question(self, understanding: Dict, state: ConversationState) -> str:
        """Generate smart follow-up question"""
        missing_info = understanding['missing_info']
        
        if 'product' in missing_info:
            products = self.vector_manager.get_all_products()
            products_text = ", ".join(products)
            prompt = f"Ask the user what product they want to return from these options: {products_text}. Be friendly and helpful."
            return self.llm_manager.generate_conversational_response(prompt)
        
        elif 'purchase_date' in missing_info:
            product = state.current_product or "your product"
            prompt = f"Ask the user when they purchased {product}. Be specific about needing the purchase date to check return eligibility."
            return self.llm_manager.generate_conversational_response(prompt)
        
        elif 'condition' in missing_info:
            product = state.current_product or "the product"
            prompt = f"Ask about the condition of {product}. Mention we need to know if it's new, used, damaged, etc. for return eligibility."
            return self.llm_manager.generate_conversational_response(prompt)
        
        else:
            prompt = "Ask the user for the missing information needed to help with their return request."
            return self.llm_manager.generate_conversational_response(prompt)
    
    def _generate_final_response(self, understanding: Dict, state: ConversationState) -> str:
        """Generate final response with policy information"""
        product = state.current_product
        
        if not product:
            return "I'd be happy to help with return policies! Please let me know what product you're asking about."
        
        # Retrieve policy from vector database
        policy_info = self._retrieve_policy_info(product)
        
        if understanding['intent'] == 'check_return_policy':
            # Just return policy information
            prompt = f"""Provide the return policy for {product} based on this information:
            Policy: {policy_info}
            
            Be helpful and informative, but don't check eligibility since the user didn't provide purchase details."""
            response = self.llm_manager.generate_conversational_response(prompt, policy_info)
            
        elif understanding['intent'] == 'check_eligibility':
            # Check eligibility with provided information
            eligibility_result = self._check_eligibility(state, policy_info)
            
            prompt = f"""Tell the user about their return eligibility for {product}:
            Policy: {policy_info}
            User's situation: Purchased {state.purchase_date or 'unknown'}, Condition: {state.product_condition or 'unknown'}
            Eligibility: {eligibility_result}
            
            Be clear and helpful, providing all relevant details."""
            response = self.llm_manager.generate_conversational_response(prompt, policy_info)
            
            # Reset conversation state after providing final answer
            state.reset()
        
        else:
            # General question
            prompt = f"""Answer the user's general question about {product} returns based on this policy:
            {policy_info}
            
            Be friendly and informative."""
            response = self.llm_manager.generate_conversational_response(prompt, policy_info)
        
        return response
    
    def _retrieve_policy_info(self, product: str) -> str:
        """Retrieve policy information from vector database"""
        try:
            relevant_docs = self.retriever.get_relevant_documents(product)
            if relevant_docs:
                # Combine all relevant documents
                policy_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                return policy_text
            else:
                return f"Standard return policy applies to {product}. Typically 30 days return period with original condition required."
        except Exception as e:
            logger.error(f"Error retrieving policy: {e}")
            return f"Return policy information for {product}. Standard conditions apply."
    
    def _check_eligibility(self, state: ConversationState, policy_info: str) -> str:
        """Check if return is eligible based on policy and user information"""
        if not state.purchase_date:
            return "Cannot determine eligibility without purchase date."
        
        try:
            # Calculate days since purchase
            purchase_date = datetime.strptime(state.purchase_date, "%Y-%m-%d")
            days_since_purchase = (datetime.now() - purchase_date).days
            
            # Extract return period from policy
            return_days = 30  # default
            if "30 days" in policy_info:
                return_days = 30
            elif "15 days" in policy_info:
                return_days = 15
            elif "45 days" in policy_info:
                return_days = 45
            elif "60 days" in policy_info:
                return_days = 60
            elif "90 days" in policy_info:
                return_days = 90
            
            # Check time eligibility
            if days_since_purchase > return_days:
                return f"Not eligible - purchased {days_since_purchase} days ago (beyond {return_days}-day return period)"
            
            # Check condition eligibility
            condition = state.product_condition.lower()
            policy_lower = policy_info.lower()
            
            if "damaged" in condition and ("no damage" in policy_lower or "undamaged" in policy_lower):
                return "Not eligible - product is damaged but policy requires no damage"
            elif "used" in condition and "unopened" in policy_lower:
                return "Not eligible - product has been used but policy requires unopened items"
            elif "worn" in condition and "unworn" in policy_lower:
                return "Not eligible - clothing has been worn but policy requires unworn items"
            
            return f"Eligible for return - within {return_days}-day period and meets conditions"
            
        except Exception as e:
            logger.error(f"Eligibility check error: {e}")
            return "Eligibility could not be determined due to missing information."

# Initialize the application
def initialize_app():
    """Initialize the vector store and agent"""
    logger.info("Initializing Product Return Agent...")
    
    # Create comprehensive product data
    product_data = [
        {
            "product_name": "Laptop",
            "return_period": "30 days",
            "conditions": "Must be in original packaging, no physical damage, all accessories included",
            "special_notes": "Software issues are covered under warranty, not return policy. Opened software cannot be returned."
        },
        {
            "product_name": "Smartphone", 
            "return_period": "15 days",
            "conditions": "Original packaging, no scratches or damage, all accessories included",
            "special_notes": "Activated phones cannot be returned. Must have original factory settings."
        },
        {
            "product_name": "Headphones",
            "return_period": "45 days",
            "conditions": "Unopened or defective units only. Original packaging required.",
            "special_notes": "Defective items can be exchanged within 1 year under warranty."
        },
        {
            "product_name": "Clothing",
            "return_period": "60 days",
            "conditions": "Tags attached, unworn, unwashed, original packaging",
            "special_notes": "Final sale items cannot be returned. Undergarments are non-returnable for hygiene reasons."
        },
        {
            "product_name": "Books",
            "return_period": "30 days", 
            "conditions": "No damage, writing, or torn pages. Original condition required.",
            "special_notes": "Digital books cannot be returned after download. Audio books have same return policy."
        },
        {
            "product_name": "Furniture",
            "return_period": "90 days",
            "conditions": "Assembly required items must be unassembled. No damage or stains.",
            "special_notes": "Shipping fees may apply for returns. Custom furniture is non-returnable."
        },
        {
            "product_name": "Tablet",
            "return_period": "30 days",
            "conditions": "Original packaging, no damage, all accessories included",
            "special_notes": "Screen protectors and cases must be unopened for full refund."
        },
        {
            "product_name": "Camera",
            "return_period": "30 days", 
            "conditions": "Original packaging, no damage, shutter count under 1000",
            "special_notes": "Memory cards and additional accessories must be unopened."
        }
    ]
    
    # Convert to documents
    documents = []
    for product in product_data:
        doc_text = f"""
        Product: {product['product_name']}
        Return Period: {product['return_period']}
        Conditions: {product['conditions']}
        Special Notes: {product['special_notes']}
        """
        documents.append(Document(
            page_content=doc_text,
            metadata={
                "product": product['product_name'],
                "return_period": product['return_period'],
                "type": "return_policy"
            }
        ))
    
    # Initialize vector store
    vector_manager = VectorStoreManager()
    vector_store = vector_manager.initialize_vector_store(documents)
    retriever = vector_manager.get_retriever(k=3)
    
    # Initialize agent
    agent = SmartProductReturnAgent(retriever)
    
    logger.info("✅ Agent initialized successfully!")
    return agent, vector_manager

# Initialize the agent
smart_agent, vector_manager = initialize_app()

# Get available products for welcome message
available_products = vector_manager.get_all_products()
welcome_message = f"""🤖 **Welcome to ReturnBot!** 

I'm your AI-powered Product Return Assistant, here to help you with return policies and eligibility checks.

**📦 Available Products I Can Help With:**
{', '.join(available_products)}

**💬 How I Can Assist You:**
• Check return policies for specific products
• Determine if you're eligible to return an item
• Explain return conditions and timeframes
• Answer any return-related questions

**🎯 Just tell me what product you're interested in, and I'll guide you through the process!**"""

# Flask Routes
@app.route('/')
def index():
    """Render the main chat interface"""
    # Initialize session
    if 'conversation_state' not in session:
        session['conversation_state'] = ConversationState().to_dict()
    if 'chat_history' not in session:
        session['chat_history'] = [{'sender': 'agent', 'message': welcome_message, 'timestamp': datetime.now().isoformat()}]
    
    return render_template('index.html', products=available_products, welcome_message=welcome_message)

@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle user messages"""
    user_message = request.json.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    # Get current conversation state from session
    conversation_state = ConversationState.from_dict(session.get('conversation_state', {}))
    chat_history = session.get('chat_history', [])
    
    # Process message with smart agent
    try:
        result = smart_agent.process_message(user_message, conversation_state)
        
        # Update session
        session['conversation_state'] = result['conversation_state']
        session['chat_history'] = conversation_state.conversation_history
        
        return jsonify({
            'response': result['response'],
            'is_followup': result['is_followup'],
            'understanding': result.get('understanding', {}),
            'chat_history': conversation_state.conversation_history
        })
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({'error': 'Sorry, I encountered an error processing your message.'}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear chat history and reset conversation"""
    session['conversation_state'] = ConversationState().to_dict()
    session['chat_history'] = [{'sender': 'agent', 'message': welcome_message, 'timestamp': datetime.now().isoformat()}]
    
    return jsonify({'success': True, 'welcome_message': welcome_message})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    """Get current chat history"""
    chat_history = session.get('chat_history', [])
    return jsonify({'chat_history': chat_history})

@app.route('/get_products', methods=['GET'])
def get_products():
    """Get available products"""
    return jsonify({'products': available_products})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
