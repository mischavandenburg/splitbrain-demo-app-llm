from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import logging
import uuid
from contextlib import asynccontextmanager
import uvicorn
import anthropic
from ollama import Client


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("api.log")],
)

logger = logging.getLogger(__name__)


# Request validation model
class QueryRequest(BaseModel):
    message: str


# Create custom logging middleware
class RequestLoggingMiddleware:
    async def __call__(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Log request details
        logger.info(f"Request started - ID: {request_id}")
        logger.info(f"Request details - Method: {request.method}, URL: {request.url}")

        try:
            # Get request body
            body = await request.body()
            if body:
                logger.info(f"Request body - ID: {request_id}, Body: {body.decode()}")
        except Exception as e:
            logger.error(
                f"Error reading request body - ID: {request_id}, Error: {str(e)}"
            )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            logger.info(
                f"Request completed - ID: {request_id}, "
                f"Status: {response.status_code}, "
                f"Processing Time: {process_time:.4f}s"
            )
            return response

        except Exception as e:
            logger.error(f"Request failed - ID: {request_id}, Error: {str(e)}")
            raise


class APICallLogger:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(f"api_call.{service_name}")

    @asynccontextmanager
    async def log_api_call(self, endpoint: str):
        call_id = str(uuid.uuid4())
        start_time = time.time()

        self.logger.info(
            f"API call started - Service: {self.service_name}, ID: {call_id}"
        )
        self.logger.info(f"API endpoint: {endpoint}")

        try:
            yield call_id
            process_time = time.time() - start_time
            self.logger.info(
                f"API call completed - Service: {self.service_name}, "
                f"ID: {call_id}, Duration: {process_time:.4f}s"
            )
        except Exception as e:
            process_time = time.time() - start_time
            self.logger.error(
                f"API call failed - Service: {self.service_name}, "
                f"ID: {call_id}, Duration: {process_time:.4f}s, Error: {str(e)}"
            )
            raise


# Initialize FastAPI app
app = FastAPI(
    title="AI Models API",
    description="API for interacting with various AI models",
    version="1.0.0",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(RequestLoggingMiddleware())

# Create logger for service
claude_logger = APICallLogger("claude")
local_logger = APICallLogger("local")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Service is running"}


@app.post("/query")
async def query_models(request: QueryRequest):
    """
    Query AI models with a message
    """
    logger.info(f"Received query request with message length: {len(request.message)}")

    async def query_claude():
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("Missing ANTHROPIC_API_KEY environment variable")
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY environment variable not set",
            )

        async with claude_logger.log_api_call("Claude API") as call_id:
            start = time.time()
            try:
                client = anthropic.Anthropic(api_key=api_key)

                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0,
                    system="You are a helpful AI assistant.",
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": request.message}],
                        }
                    ],
                )

                print(message.content)

                logger.debug(f"Message content0: {message.content[0]}")

                text_response = (
                    message.content[0].text if message.content else "No response"
                )

                result = {
                    "message": text_response,
                    "latency": time.time() - start,
                }
                logger.info(
                    f"Claude API call successful - ID: {call_id}, Latency: {result['latency']:.4f}s"
                )
                return result

            except Exception as e:
                logger.error(f"Claude API call failed - ID: {call_id}, Error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    async def query_local():
        async with local_logger.log_api_call("Local API") as call_id:
            start = time.time()
            try:
                client = Client(
                    host="http://llm:11434",
                )
                response = client.chat(
                    # model="deepseek-r1:7b",
                    model="llama3.2:1b",
                    messages=[
                        {
                            "role": "user",
                            "content": request.message,
                        },
                    ],
                )

                text_response = response.message.content

                logger.info(f"Local AI Message content: {text_response}")

                result = {
                    "message": text_response,
                    "latency": time.time() - start,
                }
                logger.info(
                    f"Local API call successful - ID: {call_id}, Latency: {result['latency']:.4f}s"
                )
                return result

            except Exception as e:
                logger.error(f"Claude API call failed - ID: {call_id}, Error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    try:
        start_time = time.time()
        logger.info("Starting API call")

        claude_result = await query_claude()
        local_result = await query_local()

        total_time = time.time() - start_time
        logger.info(f"API call completed in {total_time:.4f}s")

        return {"claude": claude_result, "local": local_result}

    except Exception as e:
        logger.error(f"Error in query_models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
