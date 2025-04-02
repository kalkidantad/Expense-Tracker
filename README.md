## Install Dependencies:

pip install google-generativeai qdrant-client pydantic numpy python-dateutil

The system combines:

-Natural language understanding (Gemini)
-Vector similarity search (Qdrant)
- Filtering (date ranges, categories)
-Structured data handling (Pydantic)

## Core Components
1. Vector Database (Qdrant) :
   - Stores expense records with embeddings
   - Enables efficient similarity searches
   - Persistent storage for data durability

2. Embedding Model (Gemini):
   - Generates 768-dimensional embeddings
   - Handles both document and query embeddings
   - Supports semantic understanding of expenses

3. Pydantic Model:
   - Expense: Data structure for expense records
   - ExpenseQuery: Validated query parameters

## Error Handling
The system includes comprehensive error handling for:

1.Invalid queries
2.Missing required fields
3.Embedding generation failures
4.Database operations
5.CSV export issues
