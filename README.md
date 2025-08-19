
# Chatbot API

API สำหรับ Chatbot ด้วย Fast API ด้วย Llama 3


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

DATABASE_URL="postgresql+psycopg2://lazymodthai:chatbot1234@localhost:5432/chatbot_db"
COLLECTION_NAME="chatbot_db"


## Running Tests

To run tests, run the following command

```bash
  uvicorn app.main:app --reload
```

