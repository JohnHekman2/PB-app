"""
LLM analysis service for PDF document analysis and conclusion generation.
No Streamlit dependencies.

Functions accept ai_provider as parameter instead of reading from session state.
Supports optional token usage tracking for rate limiting and cost analysis.
"""

import os
import tempfile
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.token_tracker import TokenUsageTracker


def analyze_local_pdf(
    uploaded_file,
    client,
    system_prompt,
    impact_prompt,
    ai_provider: str = "Interne OpenAI",
    token_tracker: Optional["TokenUsageTracker"] = None
):
    """
    Analyze a PDF file and extract impact analysis using the provided LLM client.
    
    Args:
        uploaded_file: Uploaded PDF file (file-like object with getvalue())
        client: OpenAI client instance
        system_prompt: System message for the LLM
        impact_prompt: User prompt template containing {context} placeholder
        ai_provider: AI provider selection ("Interne OpenAI" or "Mijn Gemini")
        token_tracker: Optional TokenUsageTracker for logging token consumption
        
    Returns:
        Analyzed text from the LLM or error message
        
    Raises:
        RuntimeError: If PDF analysis fails
    """
    from langchain_community.document_loaders import PyPDFLoader
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        prompt_content = impact_prompt.format(context=full_text)
        
        # Determine model based on provider
        model_name = "gemini-2.5-flash" if ai_provider == "Mijn Gemini" else "gpt-5-mini"

        # Estimate input tokens
        if token_tracker:
            input_tokens = token_tracker.estimate_tokens(
                system_prompt + prompt_content,
                model=model_name
            )
        else:
            input_tokens = 0

        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content}
            ],
            temperature=1 
        )
        
        response_text = response.choices[0].message.content
        
        # Estimate output tokens and log usage
        if token_tracker:
            output_tokens = token_tracker.estimate_tokens(response_text, model=model_name)
            token_tracker.log_token_usage(
                provider=ai_provider.lower().replace(" ", "_"),
                operation="analyze_pdf",
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        
        return response_text

    except Exception as e:
        raise RuntimeError(f"Fout bij analyseren Omgevingsvisie: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_conclusion_paragraph(
    topic: str,
    impact_analyse: str,
    client,
    conclusion_prompt_template,
    ai_provider: str = "Interne OpenAI",
    token_tracker: Optional["TokenUsageTracker"] = None
) -> str:
    """
    Generate a single conclusion paragraph for a specific topic based on impact analysis.
    
    Args:
        topic: Topic for which to generate conclusion
        impact_analyse: Impact analysis text to base conclusion on
        client: OpenAI client instance
        conclusion_prompt_template: Template with {topic} and {impact_analyse} placeholders
        ai_provider: AI provider selection ("Interne OpenAI" or "Mijn Gemini")
        token_tracker: Optional TokenUsageTracker for logging token consumption
        
    Returns:
        Generated conclusion text or error message
        
    Raises:
        ValueError: If client is not available
        RuntimeError: If conclusion generation fails
    """
    if not client:
        raise ValueError(f"OpenAI client is not available for topic '{topic}'")
    
    prompt_content = conclusion_prompt_template.format(
        topic=topic,
        impact_analyse=impact_analyse
    )
    
    try:
        # Determine model based on provider
        model_name = "gemini-2.5-flash" if ai_provider == "Mijn Gemini" else "gpt-5-mini"

        # Estimate input tokens
        if token_tracker:
            system_msg = "Je bent een expert in ecologie en ruimtelijke ordening."
            input_tokens = token_tracker.estimate_tokens(
                system_msg + prompt_content,
                model=model_name
            )
        else:
            input_tokens = 0

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Je bent een expert in ecologie en ruimtelijke ordening."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=1 
        )
        
        response_text = response.choices[0].message.content
        
        # Estimate output tokens and log usage
        if token_tracker:
            output_tokens = token_tracker.estimate_tokens(response_text, model=model_name)
            token_tracker.log_token_usage(
                provider=ai_provider.lower().replace(" ", "_"),
                operation="generate_conclusion",
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        
        return response_text
    except Exception as e:
        raise RuntimeError(f"Fout bij het genereren van conclusie voor '{topic}': {str(e)}")

