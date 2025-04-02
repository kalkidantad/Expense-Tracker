import google.generativeai as genai
from pydantic import BaseModel, model_validator
from typing import List, Optional
from qdrant_client import QdrantClient  # Vector database
from qdrant_client.models import PointStruct, Distance, VectorParams
from datetime import datetime, timedelta, date
import csv
import uuid
import time
import numpy as np

# Configure Gemini
genai.configure(api_key="gemini-api")
model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize Qdrant vector database
EMBEDDING_SIZE = 768  # Updated to match embedding-001 output
qdrant_client = QdrantClient(path="qdrant_storage")  # Persistent storage
collection_name = "expenses"

# Create collection if not exists
try:
    qdrant_client.get_collection(collection_name)
except:
    qdrant_client.create_collection(
    collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),  
)



class Expense(BaseModel):
    id: str
    amount: float
    category: str
    description: str
    date: str
    embedding: Optional[List[float]] = None
    similarity: Optional[float] = None

class ExpenseQuery(BaseModel):
    query: Optional[str] = None
    date_range: Optional[List[str]] = None
    category: Optional[str] = None

    @model_validator(mode='before')
    def check_at_least_one_field(cls, values):
        if not any(values.get(field) for field in ['query', 'date_range', 'category']):
            raise ValueError("At least one of query, date_range, or category must be provided")
        return values



def add_expense(amount: float, category: str, description: str) -> str:
    """Adds a new expense to the tracker"""
    expense_id = str(uuid.uuid4())
    date = datetime.now().isoformat()

    # Generate embedding for semantic search
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=f"{amount} {category} {description}",
        task_type="retrieval_document"
    )["embedding"]

    # embedding = embedding[:384]  
    
    expense = Expense(
        id=expense_id,
        amount=amount,
        category=category,
        description=description,
        date=date,
        embedding=embedding
    )
    
    # Store in Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=expense_id,
                vector=embedding,
                payload=expense.model_dump()  # Fixed
            )
        ]
    )

    if embedding is None or len(embedding) == 0:
        raise ValueError("Failed to generate embedding for the expense.")

    
    return f"Expense added: {amount} for {category} - {description}"

"""
    Filters and retrieves expense records from Qdrant vector database

"""
def query_expenses(query: ExpenseQuery) -> List[Expense]:
    try:
        # Get all records properly
        records, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        # converts raw records into Expense pydantic objects
        expenses = [Expense(**record.payload) for record in records if isinstance(record.payload, dict)]

        #debugging to check if embeddings exist
        # print("Checking stored expenses:")
        # for e in expenses[:5]:  # Print a few for inspection
        #     print(f"Expense: {e.amount}, {e.category}, Embedding Exists: {e.embedding is not None}")

        
        # matching the given 3 query parameters and apply filters

        # Convert date strings to datetime objects
        if query.date_range:
            start_date = datetime.strptime(query.date_range[0], "%Y-%m-%d")
            end_date = datetime.strptime(query.date_range[1], "%Y-%m-%d")

            # Convert stored expense dates to datetime for proper comparison
            expenses = [
                e for e in expenses if start_date <= datetime.fromisoformat(e.date) <= end_date
            ]
        
        if query.category:
            category_embedding = genai.embed_content(
                model="models/embedding-001",
                content=query.category,
                task_type="retrieval_query"
            )["embedding"]

            # Compute similarity scores for each stored category
            similarity_scores = {
                e.category: np.dot(category_embedding, e.embedding) /
                (np.linalg.norm(category_embedding) * np.linalg.norm(e.embedding)) for e in expenses
            }

            # Select categories with similarity above threshold
            matched_categories = [cat for cat, score in similarity_scores.items() if score > 0.6]

            # Filter expenses based on matched categories
            expenses = [e for e in expenses if e.category in matched_categories]


        # Semantic search filtering
        if query.query:

            # Augment query for better semantic understanding to catch synonyms
            # Single-word queries need expansion
            if " " not in query.query:  # Only augment if query is a single word
                    augmented_query = f"{query.query} related terms and synonyms"
            else:
                    augmented_query = query.query

            query_embedding = genai.embed_content(
                model="models/embedding-001",
                content=augmented_query,
                task_type="retrieval_query"
            )["embedding"]

            # debugging the validity of embedding
            print(f"Query embedding: {query_embedding[:15]}... (Length: {len(query_embedding)})")


            def cosine_similarity(vec1, vec2):
                return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

            # Score all expenses against the query
            scored_expenses = []
            for expense in expenses:
                similarity = cosine_similarity(expense.embedding, query_embedding)
                scored_expenses.append((expense, similarity))
            
            print(f"Top 5 similarity scores: {sorted([s[1] for s in scored_expenses], reverse=True)[:5]}")

            # Dynamic threshold (adjust based on score distribution)
            if scored_expenses:
                avg_score = sum(s[1] for s in scored_expenses) / len(scored_expenses)
                threshold = 0.6  # Increase for stricter filtering
                # threshold = max(0.3, avg_score * 0.8)  # Adaptive minimum

                # Apply similarity filtering
                scored_expenses = [e for e in scored_expenses if e[1] >= threshold]
                scored_expenses.sort(key=lambda x: x[1], reverse=True)
                expenses = [e[0] for e in scored_expenses]

            

        
        return expenses


        #     # Calculate similarity scores to measure how closely an expense matches a user's query
        #     expense_list = []
        #     for expense in expenses:
        #         similarity_score = np.dot(expense.embedding, query_embedding)
        #         if similarity_score > 0.5:  # Threshold
        #             expense_list.append({"expense": expense, "similarity": similarity_score})

        #     # Sort by similarity
        #     expense_list.sort(key=lambda e: e["similarity"], reverse=True)
        #     expenses = [e["expense"] for e in expense_list]
    
            
        # return expenses[:4]  if expenses else []
        
    except Exception as e:
        print(f"Query error: {str(e)}")
        return []



def get_spending_summary() -> dict:
    """Fixed summary function"""
    try:
        all_records = list(qdrant_client.scroll(collection_name=collection_name))
        expenses = [Expense(**record.payload) for record in all_records[0]]  # Fix payload access
        
        summary = {}
        for expense in expenses:
            summary[expense.category] = summary.get(expense.category, 0) + expense.amount
        
        return summary
    except Exception as e:
        print(f"Summary error: {str(e)}")
        return {}



def get_last_week_range() -> List[str]:
    today = date.today()
    start = (today - timedelta(days=today.weekday() + 7)).strftime("%Y-%m-%d")
    end = (today - timedelta(days=today.weekday() + 1)).strftime("%Y-%m-%d")
    return [start, end]

tools = [
    {
        "function_declarations": [
            {
                "name": "add_expense",
                "description": "Add a new expense to the tracker",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "amount": {"type_": "NUMBER", "description": "The amount spent"},
                        "category": {"type_": "STRING", "description": "Category of the expense"},
                        "description": {"type_": "STRING", "description": "Description of the expense"}
                    },
                    "required": ["amount", "category"]
                }
            },
            {
                "name": "query_expenses",
                "description": "Query expenses by date range, category, or description",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "query": {"type_": "STRING", "description": "Optional search text"},
                        "date_range": {
                            "type_": "ARRAY",
                            "items": {"type_": "STRING"},
                            "description": "Optional date range [start,end] in YYYY-MM-DD"
                        },
                        "category": {"type_": "STRING", "description": "Optional category filter"}
                    }
                }
            },
            {
                "name": "get_spending_summary",
                "description": "Get a summary of spending by category",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "time_period": {
                            "type_": "STRING",
                            "description": "Optional time period filter (today, week, month)"
                        }
                    }
                }
            },
            {
                "name": "get_system_time",
                "description": "Get the current system time",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "timezone": {
                            "type_": "STRING",
                            "description": "Optional timezone (e.g., 'UTC', 'EST')"
                        }
                    }
                }
            },
            {
                "name": "export_expenses_to_csv",
                "description": "Exports all expenses to a CSV file",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "filename": {
                            "type_": "STRING",
                            "description": "Output filename (default: expenses.csv)"
                        }
                    }
                }
            }
        ]
    }
]

def handle_user_query(user_input: str):
    chat = model.start_chat()
    
    response = chat.send_message(
        f"""You are an expense tracking assistant. Help the user with their request:
        
        User: {user_input}
        
        Respond with a function call if appropriate.""",
        tools=tools
    )
    
    # Check for function calls in the response
    for part in response.parts:
        if hasattr(part, 'function_call'):
            function_name = part.function_call.name
            args = dict(part.function_call.args)
            
            if function_name == "add_expense":
                return add_expense(**args)
            elif function_name == "query_expenses":
                expenses = query_expenses(ExpenseQuery(**args))

                if not expenses:
                    return "Sorry, I couldn't find any matching purchases."
                

                return "\n".join([f"${e.amount} - {e.category} ({e.description}) - {e.date}" for e in expenses])
            
            elif function_name == "get_spending_summary":
                # Ignore the dummy parameter
                summary = get_spending_summary()
                return "\n".join([f"{category}: ${amount}" for category, amount in summary.items()])
            
            elif function_name == "get_spending_summary":
                summary = get_spending_summary()
                if args.get('time_period'):
                    today = date.today()
                    if args['time_period'] == "today":
                        expenses = query_expenses(ExpenseQuery(
                            date_range=[today.strftime("%Y-%m-%d")]*2
                        ))
                        summary = {e.category: e.amount for e in expenses}
                return format_summary(summary)
                
            elif function_name == "get_system_time":
                tz = args.get('timezone', 'local')
                return get_system_time(tz)
                
            elif function_name == "export_expenses_to_csv":
                filename = args.get('filename', 'expenses.csv')
                return export_expenses_to_csv(filename)
    
    return response.text


def export_expenses_to_csv(filename: str = "expenses.csv") -> str:
    """Exports all expenses to a CSV file with proper error handling"""
    try:
        # Get all records with proper error handling
        records, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            return "No expenses found to export"
            
        # Prepare data
        expenses = []
        for record in records:
            try:
                expense = Expense(**record.payload)
                expenses.append([
                    expense.id,
                    expense.amount,
                    expense.category,
                    expense.description,
                    expense.date
                ])
            except Exception as e:
                print(f"Skipping malformed record: {str(e)}")
                continue
        
        # Write to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Amount", "Category", "Description", "Date"])
            writer.writerows(expenses)
            
        return f"Successfully exported {len(expenses)} expenses to {filename}"
        
    except Exception as e:
        return f"Export failed: {str(e)}"
    
# Helper functions


def format_expenses(expenses: List[Expense]) -> str:
    if not expenses:
        return "No matching expenses found"
    return "\n".join(
        f"${e.amount:.2f} - {e.category} ({e.description}) - {e.date[:10]}"
        for e in expenses
    )

def format_summary(summary: dict) -> str:
    if not summary:
        return "No expenses recorded yet"
    return "\n".join(
        f"{cat}: ${amt:.2f}" for cat, amt in summary.items()
    )

def get_system_time(timezone: str = 'local') -> str:
    now = datetime.now()
    if timezone.lower() != 'local':
        # Add timezone conversion logic here
        pass
    return now.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    print("Expense Tracker Assistant (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            response = handle_user_query(user_input)
            print("Assistant:", response)
        except Exception as e:
            print("Error:", str(e))