from flask import Flask, render_template, request, jsonify, session
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# LangChain and Vector DB imports
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production!

# Initialize components
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

# Free LLM Manager (Rule-based for simplicity)
class FreeLLMManager:
    def generate_response(self, prompt, context=""):
        """Simple rule-based response generator"""
        prompt_lower = prompt.lower()
        context_lower = context.lower()
        
        # Extract product type
        product_type = ""
        for product in ["laptop", "smartphone", "phone", "headphone", "clothing", "book", "furniture"]:
            if product in prompt_lower or product in context_lower:
                product_type = product
                break
        
        # Simple response logic
        if product_type == "laptop":
            return "Laptops can be returned within 30 days if in original condition with all accessories."
        elif product_type in ["smartphone", "phone"]:
            return "Smartphones have a 15-day return policy. Conditions: original packaging, no scratches or damage."
        elif product_type == "headphone":
            return "Headphones can be returned within 45 days if unopened or defective."
        elif product_type == "clothing":
            return "Clothing can be returned within 60 days if tags are attached and unworn."
        elif product_type == "book":
            return "Books can be returned within 30 days if undamaged and without writing."
        else:
            return "I can help you with return policies for laptops, smartphones, headphones, clothing, and books. Which product are you asking about?"

# Conversation State
@dataclass
class ConversationState:
    current_product: str = ""
    purchase_date: str = ""
    product_condition: str = ""
    waiting_for_info: str = ""  # "product", "purchase_date", "condition", "none"
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

# Enhanced Product Return Agent
class ProductReturnAgent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm_manager = FreeLLMManager()
        
    def process_message(self, user_message: str, conversation_state: ConversationState) -> Dict[str, Any]:
        """Process user message and return response"""
        print(f"Processing message: {user_message}")
        
        # Update conversation state based on user input
        self._update_conversation_state(user_message, conversation_state)
        
        # Determine what information we need
        needs = self._determine_information_needs(conversation_state)
        
        # Generate appropriate response
        if needs['waiting_for'] != 'none':
            response = self._generate_followup_question(needs['waiting_for'], conversation_state)
            is_followup = True
        else:
            response = self._generate_final_answer(conversation_state)
            is_followup = False
            # Reset for next conversation
            conversation_state.waiting_for_info = 'none'
        
        return {
            'response': response,
            'is_followup': is_followup,
            'conversation_state': conversation_state.to_dict()
        }
    
    def _update_conversation_state(self, user_message: str, state: ConversationState):
        """Update conversation state based on user input"""
        user_message_lower = user_message.lower()
        
        # Extract product type
        if not state.current_product:
            product = self._extract_product_type(user_message)
            if product:
                state.current_product = product
        
        # Extract purchase date
        if not state.purchase_date:
            purchase_date = self._extract_purchase_date(user_message)
            if purchase_date:
                state.purchase_date = purchase_date
        
        # Extract condition
        if not state.product_condition:
            condition = self._extract_condition(user_message)
            if condition:
                state.product_condition = condition
    
    def _extract_product_type(self, query: str) -> str:
        """Extract product type from user query"""
        query_lower = query.lower()
        product_mapping = {
            "laptop": ["laptop", "computer", "macbook"],
            "smartphone": ["smartphone", "phone", "iphone", "android"],
            "headphones": ["headphone", "earphone", "earbud"],
            "clothing": ["clothing", "shirt", "pants", "dress", "jacket"],
            "books": ["book", "novel", "textbook"],
            "furniture": ["furniture", "chair", "table", "sofa"]
        }
        
        for product, keywords in product_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                return product
        return ""
    
    def _extract_purchase_date(self, query: str) -> str:
        """Extract purchase date from query"""
        query_lower = query.lower()
        
        # Simple date patterns
        now = datetime.now()
        if "today" in query_lower or "just" in query_lower:
            return now.strftime("%Y-%m-%d")
        elif "yesterday" in query_lower:
            return (now - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "week" in query_lower:
            if "last" in query_lower or "1" in query_lower:
                return (now - timedelta(weeks=1)).strftime("%Y-%m-%d")
            elif "2" in query_lower:
                return (now - timedelta(weeks=2)).strftime("%Y-%m-%d")
            elif "3" in query_lower:
                return (now - timedelta(weeks=3)).strftime("%Y-%m-%d")
        elif "month" in query_lower:
            if "1" in query_lower or "one" in query_lower:
                return (now - timedelta(days=30)).strftime("%Y-%m-%d")
            elif "2" in query_lower:
                return (now - timedelta(days=60)).strftime("%Y-%m-%d")
        
        # Try to extract specific date patterns
        import re
        date_pattern = r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'
        match = re.search(date_pattern, query)
        if match:
            return f"20{match.group(3)}-{match.group(2)}-{match.group(1)}"
        
        return ""
    
    def _extract_condition(self, query: str) -> str:
        """Extract product condition from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["broken", "damaged", "cracked", "not working", "not work"]):
            return "damaged"
        elif any(word in query_lower for word in ["good", "perfect", "like new", "excellent", "mint"]):
            return "good"
        elif any(word in query_lower for word in ["used", "opened", "worn", "normal wear"]):
            return "used"
        elif any(word in query_lower for word in ["new", "unopened", "sealed"]):
            return "new"
        
        return ""
    
    def _determine_information_needs(self, conversation_state: ConversationState) -> Dict[str, Any]:
        """Determine what information we're missing"""
        needs = {
            "needs_retrieval": True,
            "waiting_for": "none"
        }
        
        # Check if we have a product
        if not conversation_state.current_product:
            needs["waiting_for"] = "product"
            return needs
        
        # Check if we have purchase date
        if not conversation_state.purchase_date:
            needs["waiting_for"] = "purchase_date"
            return needs
        
        # Check if we have condition
        if not conversation_state.product_condition:
            needs["waiting_for"] = "condition"
            return needs
        
        # We have all information
        needs["waiting_for"] = "none"
        return needs
    
    def _generate_followup_question(self, waiting_for: str, conversation_state: ConversationState) -> str:
        """Generate appropriate follow-up question"""
        if waiting_for == "product":
            return "I'd be happy to help with your return! Could you please tell me what product you'd like to return? (e.g., laptop, smartphone, clothing, headphones, books, furniture)"
        
        elif waiting_for == "purchase_date":
            product = conversation_state.current_product
            return f"Thank you! To check the return eligibility for your {product}, could you please tell me when you purchased it? (e.g., '2 weeks ago', 'yesterday', 'March 15th', '2024-01-15')"
        
        elif waiting_for == "condition":
            product = conversation_state.current_product
            return f"Great! Finally, could you describe the condition of your {product}? (e.g., 'like new', 'used but working', 'damaged', 'unopened')"
        
        else:
            return "I need some additional information to help with your return. Could you please provide more details?"
    
    def _generate_final_answer(self, conversation_state: ConversationState) -> str:
        """Generate final answer with eligibility determination"""
        product = conversation_state.current_product
        purchase_date_str = conversation_state.purchase_date
        condition = conversation_state.product_condition
        
        # Calculate days since purchase
        try:
            if purchase_date_str:
                purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d")
                days_since_purchase = (datetime.now() - purchase_date).days
            else:
                days_since_purchase = 0
        except:
            days_since_purchase = 30  # Default assumption
        
        # Get return policy
        return_period, conditions_text = self._get_return_policy(product)
        
        # Determine eligibility
        is_eligible, reason = self._check_eligibility(
            days_since_purchase, return_period, condition, conditions_text
        )
        
        # Generate response
        if is_eligible:
            response = f"✅ **Yes, you CAN return your {product}!**\n\n"
        else:
            response = f"❌ **No, you CANNOT return your {product}.**\n\n"
        
        response += f"**Return Policy Details:**\n"
        response += f"• Return Period: {return_period}\n"
        response += f"• Conditions: {conditions_text}\n"
        response += f"• Your purchase: {days_since_purchase} days ago\n"
        response += f"• Your product condition: {condition}\n\n"
        response += f"**Reason:** {reason}\n\n"
        
        if is_eligible:
            response += "You can proceed with the return process through our website or contact customer service."
        else:
            response += "You might want to check if your product is still under warranty for repair options."
        
        return response
    
    def _get_return_policy(self, product: str) -> tuple:
        """Get return policy for product"""
        policies = {
            "laptop": ("30 days", "Must be in original packaging, no physical damage, all accessories included"),
            "smartphone": ("15 days", "Original packaging, no scratches or damage, all accessories"),
            "headphones": ("45 days", "Unopened or defective units only"),
            "clothing": ("60 days", "Tags attached, unworn, unwashed"),
            "books": ("30 days", "No damage, writing, or torn pages"),
            "furniture": ("90 days", "Assembly required items must be unassembled")
        }
        
        return policies.get(product, ("30 days", "Standard return policy applies"))
    
    def _check_eligibility(self, days_since_purchase: int, return_period: str, condition: str, conditions_text: str) -> tuple:
        """Check if return is eligible based on criteria"""
        # Extract days from return period string
        return_days = 30  # default
        if "day" in return_period:
            try:
                return_days = int(''.join(filter(str.isdigit, return_period)))
            except:
                return_days = 30
        
        # Check time eligibility
        if days_since_purchase > return_days:
            return False, f"Your purchase was {days_since_purchase} days ago, but our return period is only {return_days} days."
        
        # Check condition eligibility
        condition_lower = condition.lower()
        conditions_lower = conditions_text.lower()
        
        if "damaged" in condition_lower and "no damage" in conditions_lower:
            return False, "The product is damaged but our policy requires no physical damage for returns."
        
        if "used" in condition_lower and "unopened" in conditions_lower:
            return False, "The product has been used but our policy requires unopened items for return."
        
        if "worn" in condition_lower and "unworn" in conditions_lower:
            return False, "The clothing has been worn but our policy requires unworn items for return."
        
        return True, "Your product meets all the return policy criteria."

# Initialize the application
def initialize_app():
    """Initialize the vector store and agent"""
    # Create sample product data
    product_data = [
        {
            "product_name": "Laptop",
            "return_period": "30 days",
            "conditions": "Must be in original packaging, no physical damage, all accessories included",
            "special_notes": "Software issues are covered under warranty, not return policy"
        },
        {
            "product_name": "Smartphone", 
            "return_period": "15 days",
            "conditions": "Original packaging, no scratches or damage, all accessories",
            "special_notes": "Opened software cannot be returned if activated"
        },
        {
            "product_name": "Headphones",
            "return_period": "45 days",
            "conditions": "Unopened or defective units only",
            "special_notes": "Defective items can be exchanged within 1 year"
        },
        {
            "product_name": "Clothing",
            "return_period": "60 days",
            "conditions": "Tags attached, unworn, unwashed",
            "special_notes": "Final sale items cannot be returned"
        },
        {
            "product_name": "Books",
            "return_period": "30 days", 
            "conditions": "No damage, writing, or torn pages",
            "special_notes": "Digital books cannot be returned after download"
        },
        {
            "product_name": "Furniture",
            "return_period": "90 days",
            "conditions": "Assembly required items must be unassembled",
            "special_notes": "Shipping fees may apply for returns"
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
            metadata={"product": product['product_name']}
        ))
    
    # Initialize vector store
    vector_manager = VectorStoreManager()
    vector_store = vector_manager.initialize_vector_store(documents)
    retriever = vector_manager.get_retriever(k=3)
    
    # Initialize agent
    agent = ProductReturnAgent(retriever)
    
    return agent

# Initialize the agent
print("Initializing Product Return Agent...")
agent = initialize_app()
print("Agent initialized successfully!")

# Flask Routes
@app.route('/')
def index():
    """Render the main chat interface"""
    # Initialize session
    if 'conversation_state' not in session:
        session['conversation_state'] = ConversationState().to_dict()
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle user messages"""
    user_message = request.json.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    # Get current conversation state from session
    conversation_state = ConversationState.from_dict(session.get('conversation_state', {}))
    chat_history = session.get('chat_history', [])
    
    # Add user message to chat history
    chat_history.append({'sender': 'user', 'message': user_message})
    
    # Process message with agent
    try:
        result = agent.process_message(user_message, conversation_state)
        
        # Add agent response to chat history
        chat_history.append({'sender': 'agent', 'message': result['response']})
        
        # Update session
        session['conversation_state'] = result['conversation_state']
        session['chat_history'] = chat_history
        
        return jsonify({
            'response': result['response'],
            'is_followup': result['is_followup'],
            'chat_history': chat_history
        })
        
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'error': 'Sorry, I encountered an error processing your message.'}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear chat history and reset conversation"""
    session['conversation_state'] = ConversationState().to_dict()
    session['chat_history'] = []
    
    return jsonify({'success': True})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    """Get current chat history"""
    chat_history = session.get('chat_history', [])
    return jsonify({'chat_history': chat_history})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)