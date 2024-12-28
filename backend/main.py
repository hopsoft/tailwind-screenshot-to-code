# Load environment variables first
from dotenv import load_dotenv

load_dotenv()


import json
import os
import traceback
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from llm import stream_openai_response
from mock import mock_completion
from image_generation import create_alt_url_mapping, generate_images
from prompts import assemble_prompt

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Useful for debugging purposes when you don't want to waste GPT4-Vision credits
# Setting to True will stream a mock response instead of calling the OpenAI API
SHOULD_MOCK_AI_RESPONSE = False


def write_logs(prompt_messages, completion):
    # Get the logs path from environment, default to the current working directory
    logs_path = os.environ.get("LOGS_PATH", os.getcwd())

    # Create run_logs directory if it doesn't exist within the specified logs path
    logs_directory = os.path.join(logs_path, "run_logs")
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    print("Writing to logs directory:", logs_directory)

    # Generate a unique filename using the current timestamp within the logs directory
    filename = datetime.now().strftime(f"{logs_directory}/messages_%Y%m%d_%H%M%S.json")

    # Write the messages dict into a new file for each run
    with open(filename, "w") as f:
        f.write(json.dumps({"prompt": prompt_messages, "completion": completion}))


@app.websocket("/generate-code")
async def stream_code_test(websocket: WebSocket):
    print("WebSocket connection attempt from:", websocket.client)
    try:
        await websocket.accept()
        print("WebSocket connection accepted")

        try:
            params = await websocket.receive_json()
            print("Received params")
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", str(e))
            await websocket.close(code=1003)  # Unsupported data
            return
        except Exception as e:
            print("Error receiving JSON:", str(e))
            await websocket.close(code=1011)  # Internal error
            return

        try:
            # Get the OpenAI API key from the request. Fall back to environment variable if not provided.
            # If neither is provided, we throw an error.
            openai_api_key = params.get("openAiApiKey") or os.environ.get("OPENAI_API_KEY")

            if not openai_api_key:
                print("OpenAI API key not found")
                await websocket.send_json({
                    "type": "error",
                    "value": "No OpenAI API key found. Please add your API key in the settings dialog or add it to backend/.env file.",
                })
                await websocket.close(code=1000)  # Normal closure
                return

            should_generate_images = params.get("isImageGenerationEnabled", True)

            print("Generating code...")
            await websocket.send_json({"type": "status", "value": "Generating code..."})

            async def process_chunk(content):
                await websocket.send_json({"type": "chunk", "value": content})

            prompt_messages = assemble_prompt(params["image"])

            # Image cache for updates so that we don't have to regenerate images
            image_cache = {}

            if params["generationType"] == "update":
                # Transform into message format
                for index, text in enumerate(params["history"]):
                    prompt_messages += [
                        {"role": "assistant" if index % 2 == 0 else "user", "content": text}
                    ]

                image_cache = create_alt_url_mapping(params["history"][-2])

            if SHOULD_MOCK_AI_RESPONSE:
                completion = await mock_completion(process_chunk)
            else:
                completion = await stream_openai_response(
                    prompt_messages,
                    api_key=openai_api_key,
                    callback=lambda x: process_chunk(x),
                )

            # Write the messages dict into a log so that we can debug later
            write_logs(prompt_messages, completion)

            if should_generate_images:
                await websocket.send_json(
                    {"type": "status", "value": "Generating images..."}
                )
                updated_html = await generate_images(
                    completion, api_key=openai_api_key, image_cache=image_cache
                )
            else:
                updated_html = completion

            await websocket.send_json({"type": "setCode", "value": updated_html})
            await websocket.send_json(
                {"type": "status", "value": "Code generation complete."}
            )
            await websocket.close(code=1000)  # Normal closure

        except WebSocketDisconnect:
            print("WebSocket disconnected by client")
        except Exception as e:
            print("Error during processing:", str(e))
            traceback.print_exc()
            try:
                await websocket.send_json({
                    "type": "error",
                    "value": "An error occurred during processing: " + str(e)
                })
            except:
                pass  # Connection might already be closed
            await websocket.close(code=1011)  # Internal error
    except Exception as e:
        print("Error in WebSocket handler:", str(e))
        traceback.print_exc()
        try:
            await websocket.close(code=1011)  # Internal error
        except:
            pass  # Connection might already be closed
