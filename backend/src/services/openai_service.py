"""
OpenAI Integration Service for AI Code Mentor
Handles AI operations including embeddings, completions, and code analysis.

Compliance:
- RULE SEC-005: API key management and secure external service integration
- RULE PERF-001: Optimized AI request handling with rate limiting
- RULE LOG-001: Structured logging for AI operations
"""

import asyncio
import time
from typing import Dict, List, Optional, Union
from uuid import UUID

import openai
import structlog
from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class OpenAIServiceError(Exception):
    """Custom exception for OpenAI service errors."""
    pass


class OpenAIService:
    """Service class for OpenAI API operations."""

    def __init__(self):
        """Initialize OpenAI service with configuration."""
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            organization=settings.OPENAI_ORG_ID
        )
        self.model = settings.OPENAI_MODEL
        self.embedding_model = settings.OPENAI_EMBEDDING_MODEL
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.temperature = settings.OPENAI_TEMPERATURE
        
        # Rate limiting - simple implementation
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 10 requests per second max

    async def _rate_limit(self):
        """Simple rate limiting to avoid hitting OpenAI limits."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()

    async def create_embeddings(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Create embeddings for given text(s).
        
        Args:
            texts: Single text string or list of text strings
            model: Optional model override
            
        Returns:
            List of embedding vectors
            
        Raises:
            OpenAIServiceError: If embedding creation fails
        """
        try:
            await self._rate_limit()
            
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Validate inputs
            if not texts or any(not text.strip() for text in texts):
                raise OpenAIServiceError("Invalid or empty text provided")
            
            model_to_use = model or self.embedding_model
            
            logger.info(
                "Creating embeddings",
                text_count=len(texts),
                model=model_to_use,
                total_chars=sum(len(text) for text in texts)
            )
            
            # Create embeddings
            response: CreateEmbeddingResponse = await self.client.embeddings.create(
                input=texts,
                model=model_to_use
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            logger.info(
                "Embeddings created successfully",
                embedding_count=len(embeddings),
                embedding_dimensions=len(embeddings[0]) if embeddings else 0,
                tokens_used=response.usage.total_tokens if response.usage else "unknown"
            )
            
            return embeddings
            
        except openai.RateLimitError as e:
            logger.error(
                "OpenAI rate limit exceeded",
                error=str(e),
                text_count=len(texts) if isinstance(texts, list) else 1
            )
            raise OpenAIServiceError(f"Rate limit exceeded: {str(e)}")
        except openai.AuthenticationError as e:
            logger.error("OpenAI authentication failed", error=str(e))
            raise OpenAIServiceError(f"Authentication failed: {str(e)}")
        except Exception as e:
            logger.error(
                "Failed to create embeddings",
                error=str(e),
                error_type=type(e).__name__
            )
            raise OpenAIServiceError(f"Embedding creation failed: {str(e)}")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate chat completion using OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override
            max_tokens: Optional max tokens override
            temperature: Optional temperature override
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Generated response text
            
        Raises:
            OpenAIServiceError: If completion fails
        """
        try:
            await self._rate_limit()
            
            # Validate messages
            if not messages:
                raise OpenAIServiceError("No messages provided")
            
            # Prepare messages with optional system prompt
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            formatted_messages.extend(messages)
            
            # Validate message format
            for msg in formatted_messages:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    raise OpenAIServiceError("Invalid message format")
                if msg['role'] not in ['system', 'user', 'assistant']:
                    raise OpenAIServiceError(f"Invalid message role: {msg['role']}")
            
            model_to_use = model or self.model
            max_tokens_to_use = max_tokens or self.max_tokens
            temperature_to_use = temperature if temperature is not None else self.temperature
            
            logger.info(
                "Creating chat completion",
                model=model_to_use,
                message_count=len(formatted_messages),
                max_tokens=max_tokens_to_use,
                temperature=temperature_to_use
            )
            
            # Create completion
            response: ChatCompletion = await self.client.chat.completions.create(
                model=model_to_use,
                messages=formatted_messages,
                max_tokens=max_tokens_to_use,
                temperature=temperature_to_use
            )
            
            # Extract response
            if not response.choices or not response.choices[0].message:
                raise OpenAIServiceError("No response generated")
            
            content = response.choices[0].message.content
            if not content:
                raise OpenAIServiceError("Empty response generated")
            
            logger.info(
                "Chat completion successful",
                response_length=len(content),
                tokens_used=response.usage.total_tokens if response.usage else "unknown",
                finish_reason=response.choices[0].finish_reason
            )
            
            return content
            
        except openai.RateLimitError as e:
            logger.error("OpenAI rate limit exceeded", error=str(e))
            raise OpenAIServiceError(f"Rate limit exceeded: {str(e)}")
        except openai.AuthenticationError as e:
            logger.error("OpenAI authentication failed", error=str(e))
            raise OpenAIServiceError(f"Authentication failed: {str(e)}")
        except Exception as e:
            logger.error(
                "Failed to create chat completion",
                error=str(e),
                error_type=type(e).__name__
            )
            raise OpenAIServiceError(f"Chat completion failed: {str(e)}")

    async def analyze_code(
        self,
        code_content: str,
        analysis_type: str = "general",
        language: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Analyze code content using AI.
        
        Args:
            code_content: The code to analyze
            analysis_type: Type of analysis (general, security, performance, etc.)
            language: Programming language (auto-detected if not provided)
            
        Returns:
            Analysis results dictionary
            
        Raises:
            OpenAIServiceError: If analysis fails
        """
        try:
            if not code_content.strip():
                raise OpenAIServiceError("No code content provided")
            
            # Prepare system prompt based on analysis type
            system_prompts = {
                "general": """
                You are an expert code reviewer and mentor. Analyze the provided code and provide:
                1. Code quality assessment
                2. Potential improvements
                3. Best practices recommendations
                4. Security considerations
                5. Performance implications
                
                Provide your analysis in a structured format with clear sections.
                """,
                "security": """
                You are a security expert. Analyze the provided code for security vulnerabilities:
                1. Identify potential security issues
                2. Explain the risks
                3. Provide remediation recommendations
                4. Rate the severity of each issue
                
                Focus specifically on security aspects.
                """,
                "performance": """
                You are a performance optimization expert. Analyze the provided code for performance:
                1. Identify performance bottlenecks
                2. Suggest optimizations
                3. Analyze time/space complexity
                4. Recommend better algorithms or data structures
                
                Focus specifically on performance optimization.
                """
            }
            
            system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
            
            # Add language context if provided
            if language:
                system_prompt += f"\n\nThe code is written in {language}."
            
            messages = [
                {
                    "role": "user",
                    "content": f"Please analyze this code:\n\n```\n{code_content}\n```"
                }
            ]
            
            logger.info(
                "Starting code analysis",
                analysis_type=analysis_type,
                language=language,
                code_length=len(code_content)
            )
            
            # Get AI analysis
            analysis_result = await self.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.1  # Lower temperature for more consistent analysis
            )
            
            return {
                "analysis_type": analysis_type,
                "language": language,
                "result": analysis_result,
                "code_length": len(code_content),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(
                "Code analysis failed",
                analysis_type=analysis_type,
                error=str(e)
            )
            raise OpenAIServiceError(f"Code analysis failed: {str(e)}")

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
            
        Raises:
            OpenAIServiceError: If response generation fails
        """
        try:
            if not question.strip():
                raise OpenAIServiceError("No question provided")
            
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
            
            # Build conversation
            messages = []
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-10:])  # Limit to last 10 messages
            
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
                question_length=len(question),
                context_length=len(context),
                history_messages=len(conversation_history) if conversation_history else 0
            )
            
            # Generate response
            response = await self.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.2  # Slightly higher for more natural responses
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Q&A response generation failed",
                error=str(e)
            )
            raise OpenAIServiceError(f"Q&A response generation failed: {str(e)}")

    async def health_check(self) -> Dict[str, any]:
        """
        Perform health check on OpenAI service.
        
        Returns:
            Health status dictionary
        """
        try:
            # Simple test with minimal token usage
            test_response = await self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return {
                "status": "healthy",
                "service": "openai",
                "test_response_length": len(test_response),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error("OpenAI health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "service": "openai",
                "error": str(e),
                "timestamp": time.time()
            }


# Global service instance
openai_service = OpenAIService()