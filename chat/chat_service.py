import os
import logging
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatService:
    def __init__(self):
        self.system_prompt = {
            "role": "system",
            "content": """You are a friendly healthcare assistant focused on blood pressure measurement. 
            Your role is to:
            - Guide users through proper blood pressure measurement procedures
            - Explain correct cuff positioning and posture
            - Provide health education related to blood pressure
            - Answer questions about hypertension and cardiovascular health
            - Remind users about measurement best practices
            
            Keep responses clear, professional, and easy to understand.
            If asked about medical conditions or treatments, remind users to consult healthcare professionals.
            *Critical*: Must response in Vietnamese language.
            """
        }

    async def chat(self, message):
        """
        Handle chat with OpenAI
        Returns: generator of response chunks
        """
        try:
            # Initialize conversation with system prompt
            messages = [self.system_prompt]
            
            # Add user message to history
            messages.append({"role": "user", "content": message})
            
            try:
                # Call OpenAI API with streaming
                stream = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                
                # Process streaming response
                current_response = []
                for chunk in stream:
                    if hasattr(chunk.choices[0], "delta") and \
                       hasattr(chunk.choices[0].delta, "content") and \
                       chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        current_response.append(content)
                        yield {
                            "content": content,
                            "is_error": False,
                            "error_message": None
                        }
                
                # Add assistant's complete response to history
                messages.append({
                    "role": "assistant", 
                    "content": "".join(current_response)
                })
                
            except Exception as e:
                yield {
                    "content": None,
                    "is_error": True,
                    "error_message": f"Error calling OpenAI API: {str(e)}"
                }
                
        except Exception as e:
            yield {
                "content": None,
                "is_error": True,
                "error_message": f"Stream error: {str(e)}"
            }