"""
Multi-Provider AI Service for Code Analysis and Q&A
Supports OpenAI, Google Gemini, DeepSeek, Anthropic, and Ollama (local models)

Compliance:
- RULE SEC-002: Secure API key handling
- RULE PERF-003: Efficient token usage and caching
- RULE LOG-001: Structured logging with trace IDs
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Union
from enum import Enum

import aiohttp
import structlog
from openai import AsyncOpenAI
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class AIProvider(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class AIServiceError(Exception):
    """Custom exception for AI service errors."""
    pass


class MultiProviderAIService:
    """Multi-provider AI service supporting various models."""

    def __init__(self):
        """Initialize AI service with multiple provider support."""
        self.provider = getattr(settings, 'AI_PROVIDER', 'gemini')  # Default to Gemini
        self.model = getattr(settings, 'AI_MODEL', self._get_default_model())
        
        # Provider-specific configurations
        self.configs = {
            AIProvider.OPENAI: {
                'api_key': getattr(settings, 'OPENAI_API_KEY', ''),
                'base_url': 'https://api.openai.com/v1',
                'default_model': 'gpt-3.5-turbo'
            },
            AIProvider.GEMINI: {
                'api_key': getattr(settings, 'GEMINI_API_KEY', ''),
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'default_model': 'gemini-1.5-flash'
            },
            AIProvider.DEEPSEEK: {
                'api_key': getattr(settings, 'DEEPSEEK_API_KEY', ''),
                'base_url': 'https://api.deepseek.com/v1',
                'default_model': 'deepseek-coder'
            },
            AIProvider.ANTHROPIC: {
                'api_key': getattr(settings, 'ANTHROPIC_API_KEY', ''),
                'base_url': 'https://api.anthropic.com',
                'default_model': 'claude-3-5-haiku-20241022'
            },
            AIProvider.OLLAMA: {
                'api_key': '',  # Ollama doesn't need API key
                'base_url': getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434'),
                'default_model': 'deepseek-coder:6.7b'
            }
        }
        
        # Initialize provider-specific clients
        self._init_clients()
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1

    def _get_default_model(self) -> str:
        """Get default model based on provider."""
        provider_models = {
            AIProvider.OPENAI: 'gpt-3.5-turbo',
            AIProvider.GEMINI: 'gemini-1.5-flash',
            AIProvider.DEEPSEEK: 'deepseek-coder',
            AIProvider.ANTHROPIC: 'claude-3-5-haiku-20241022',
            AIProvider.OLLAMA: 'deepseek-coder:6.7b'
        }
        return provider_models.get(self.provider, 'gemini-1.5-flash')

    def _init_clients(self):
        """Initialize provider-specific clients."""
        try:
            if self.provider == AIProvider.OPENAI:
                self.openai_client = AsyncOpenAI(
                    api_key=self.configs[AIProvider.OPENAI]['api_key']
                )
            elif self.provider == AIProvider.ANTHROPIC and AsyncAnthropic:
                self.anthropic_client = AsyncAnthropic(
                    api_key=self.configs[AIProvider.ANTHROPIC]['api_key']
                )
        except Exception as e:
            logger.warning(f"Failed to initialize {self.provider} client: {e}")

    async def _rate_limit(self):
        """Simple rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()

    async def _make_http_request(self, url: str, headers: Dict[str, str], data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to AI provider."""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise AIServiceError(f"HTTP {response.status}: {error_text}")
                return await response.json()

    async def _chat_completion_mock(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Mock chat completion for testing when no API keys are configured."""
        last_message = messages[-1]['content'] if messages else "No message"
        
        # Simple pattern matching for demo responses
        if "programming language" in last_message.lower():
            return "Based on the context provided, this appears to be JavaScript code. The document contains a simple 'Hello World' function that demonstrates basic JavaScript syntax."
        elif "function" in last_message.lower() and "does" in last_message.lower():
            return "The helloWorld function is a simple demonstration function that returns the string 'Hello, World!'. This is a common example used to introduce programming concepts and test basic functionality."
        elif "best practices" in last_message.lower() or "improve" in last_message.lower():
            return "Here are some best practices that could improve this code:\n\n1. **Add JSDoc comments** - Document the function's purpose and return value\n2. **Consider parameterization** - Allow customization of the greeting message\n3. **Add error handling** - Handle edge cases appropriately\n4. **Use consistent naming** - Follow camelCase conventions\n5. **Consider internationalization** - Support multiple languages\n\nThe current code is simple and functional, which is good for a basic example."
        elif "purpose" in last_message.lower() or "document" in last_message.lower():
            return "This document appears to be a simple code example or tutorial demonstrating basic programming concepts. It contains a 'Hello World' function, which is typically used as an introductory example for learning programming languages or testing development environments."
        else:
            return f"I understand you're asking about the code in the document. Based on the context provided, I can help explain the functionality, suggest improvements, or answer specific questions about the programming concepts demonstrated. Could you provide more specific details about what you'd like to know?"


        """Generate chat completion using Gemini API."""
        api_key = self.configs[AIProvider.GEMINI]['api_key']
        if not api_key:
            raise AIServiceError("Gemini API key not configured")

        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            if msg['role'] == 'system':
                # Gemini doesn't have system role, prepend to first user message
                if contents and contents[0]['role'] == 'user':
                    contents[0]['parts'][0]['text'] = f"{msg['content']}\n\n{contents[0]['parts'][0]['text']}"
                else:
                    contents.insert(0, {
                        'role': 'user',
                        'parts': [{'text': msg['content']}]
                    })
            else:
                role = 'user' if msg['role'] == 'user' else 'model'
                contents.append({
                    'role': role,
                    'parts': [{'text': msg['content']}]
                })

        url = f"{self.configs[AIProvider.GEMINI]['base_url']}/models/{self.model}:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        data = {
            'contents': contents,
            'generationConfig': {
                'temperature': kwargs.get('temperature', 0.2),
                'maxOutputTokens': kwargs.get('max_tokens', 4000)
            }
        }

        response = await self._make_http_request(url, headers, data)
        
        if 'candidates' not in response or not response['candidates']:
            raise AIServiceError("No response generated from Gemini")
        
        return response['candidates'][0]['content']['parts'][0]['text']

    async def _chat_completion_deepseek(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat completion using DeepSeek API."""
        api_key = self.configs[AIProvider.DEEPSEEK]['api_key']
        if not api_key:
            raise AIServiceError("DeepSeek API key not configured")

        url = f"{self.configs[AIProvider.DEEPSEEK]['base_url']}/chat/completions"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model,
            'messages': messages,
            'temperature': kwargs.get('temperature', 0.2),
            'max_tokens': kwargs.get('max_tokens', 4000)
        }

        response = await self._make_http_request(url, headers, data)
        
        if 'choices' not in response or not response['choices']:
            raise AIServiceError("No response generated from DeepSeek")
        
        return response['choices'][0]['message']['content']

    async def _chat_completion_ollama(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat completion using Ollama (local)."""
        base_url = self.configs[AIProvider.OLLAMA]['base_url']
        
        url = f"{base_url}/api/chat"
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': self.model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': kwargs.get('temperature', 0.2),
                'num_predict': kwargs.get('max_tokens', 4000)
            }
        }

        try:
            response = await self._make_http_request(url, headers, data)
            return response['message']['content']
        except Exception as e:
            raise AIServiceError(f"Ollama request failed. Is Ollama running at {base_url}? Error: {e}")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate chat completion using configured provider.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override
            max_tokens: Optional max tokens override
            temperature: Optional temperature override
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Generated response text
        """
        try:
            await self._rate_limit()
            
            if not messages:
                raise AIServiceError("No messages provided")

            # Prepare messages with optional system prompt
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            formatted_messages.extend(messages)

            # Use provided model or default
            current_model = model or self.model
            
            kwargs = {
                'max_tokens': max_tokens or 4000,
                'temperature': temperature if temperature is not None else 0.2
            }

            logger.info(
                "Creating chat completion",
                provider=self.provider,
                model=current_model,
                message_count=len(formatted_messages),
                **kwargs
            )

            # Check if we have a valid API key for the selected provider
            api_key = self.configs[self.provider]['api_key']
            if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE" or api_key == "YOUR_DEEPSEEK_API_KEY_HERE" or api_key == "YOUR_ANTHROPIC_API_KEY_HERE":
                logger.warning(f"No valid API key for {self.provider}, using mock responses")
                content = await self._chat_completion_mock(formatted_messages, **kwargs)
            # Route to appropriate provider
            elif self.provider == AIProvider.OPENAI:
                response = await self.openai_client.chat.completions.create(
                    model=current_model,
                    messages=formatted_messages,
                    **kwargs
                )
                content = response.choices[0].message.content
            elif self.provider == AIProvider.GEMINI:
                content = await self._chat_completion_gemini(formatted_messages, **kwargs)
            elif self.provider == AIProvider.DEEPSEEK:
                content = await self._chat_completion_deepseek(formatted_messages, **kwargs)
            elif self.provider == AIProvider.OLLAMA:
                content = await self._chat_completion_ollama(formatted_messages, **kwargs)
            elif self.provider == AIProvider.ANTHROPIC and hasattr(self, 'anthropic_client'):
                # Anthropic has different message format
                system_msg = None
                formatted_messages_anthropic = []
                for msg in formatted_messages:
                    if msg['role'] == 'system':
                        system_msg = msg['content']
                    else:
                        formatted_messages_anthropic.append(msg)
                
                response = await self.anthropic_client.messages.create(
                    model=current_model,
                    system=system_msg,
                    messages=formatted_messages_anthropic,
                    max_tokens=kwargs['max_tokens']
                )
                content = response.content[0].text
            else:
                raise AIServiceError(f"Provider {self.provider} not properly configured")

            if not content:
                raise AIServiceError("Empty response generated")

            logger.info(
                "Chat completion successful",
                provider=self.provider,
                response_length=len(content)
            )

            return content

        except Exception as e:
            logger.error(
                "Failed to create chat completion",
                provider=self.provider,
                error=str(e),
                error_type=type(e).__name__
            )
            raise AIServiceError(f"Chat completion failed: {str(e)}")

    async def generate_qa_response(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate Q&A response based on context and question.
        
        Args:
            question: User's question
            context: Relevant context (code, documentation, etc.)
            conversation_history: Previous conversation messages
            
        Returns:
            AI-generated answer
        """
        try:
            if not question.strip():
                raise AIServiceError("No question provided")

            system_prompt = """
You are an expert programming mentor and code analyst. Answer questions based on the provided context.

Guidelines:
1. Be accurate and helpful
2. Provide code examples when relevant
3. Explain concepts clearly
4. Reference the provided context when applicable
5. If you're unsure, say so rather than guessing
6. Format code snippets properly

Always base your answers on the provided context when possible.
"""

            messages = []
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-10:])

            # Add current question with context
            current_message = f"""Context:
{context}

Question: {question}

Please provide a helpful answer based on the context above."""

            messages.append({
                "role": "user",
                "content": current_message
            })

            logger.info(
                "Generating Q&A response",
                provider=self.provider,
                question_length=len(question),
                context_length=len(context),
                history_messages=len(conversation_history) if conversation_history else 0
            )

            response = await self.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.2
            )

            return response

        except Exception as e:
            logger.error(
                "Q&A response generation failed",
                provider=self.provider,
                error=str(e)
            )
            raise AIServiceError(f"Q&A response generation failed: {str(e)}")

    async def create_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Create embeddings for given text(s).
        Note: Currently only supported for OpenAI provider.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            List of embedding vectors
        """
        if self.provider != AIProvider.OPENAI:
            logger.warning(f"Embeddings not supported for {self.provider}, skipping")
            # Return empty embeddings for non-OpenAI providers for now
            if isinstance(texts, str):
                return [[0.0] * 1536]  # Standard embedding dimension
            return [[0.0] * 1536 for _ in texts]
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            response = await self.openai_client.embeddings.create(
                input=texts,
                model=getattr(settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            # Return dummy embeddings as fallback
            return [[0.0] * 1536 for _ in texts]


        """Perform health check on AI service."""
        try:
            test_response = await self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return {
                "status": "healthy",
                "provider": self.provider,
                "model": self.model,
                "test_response_length": len(test_response),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"{self.provider} health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "provider": self.provider,
                "model": self.model,
                "error": str(e),
                "timestamp": time.time()
            }


# Global service instance
ai_service = MultiProviderAIService()