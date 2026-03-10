"""
RAG (Retrieval-Augmented Generation) service for batch area analysis.
No Streamlit dependencies.

Progress tracking is handled via optional callback functions passed by the caller.
Supports optional token usage tracking for rate limiting and cost analysis.
"""

import concurrent.futures
from typing import Callable, Dict, Optional, List, Tuple, TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Local imports
from services.data_processing import parse_json_response, format_json_to_markdown

if TYPE_CHECKING:
    from services.token_tracker import TokenUsageTracker


def create_filtered_retriever(vector_store, selected_areas):
    """
    Create a retriever filtered to specific areas.
    
    Args:
        vector_store: Chroma vector store instance
        selected_areas: List of area names to retrieve documents for
        
    Returns:
        Configured retriever for the vector store
    """
    RETRIEVAL_COUNT = 15
    if not selected_areas:
        return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_COUNT})
    if len(selected_areas) == 1:
        chroma_filter = {"area_name": selected_areas[0]}
    else:
        chroma_filter = {"$or": [{"area_name": area} for area in selected_areas]}
    return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_COUNT, "filter": chroma_filter})


def get_rag_chain(_retriever, _llm, system_template):
    """
    Build a RAG chain that combines retrieval with LLM generation.
    
    Args:
        _retriever: Document retriever
        _llm: LangChain LLM instance
        system_template: System prompt template string
        
    Returns:
        Compiled RAG chain (Runnable)
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}") 
    ])
    return RunnableParallel({
        "context": _retriever,
        "question": RunnablePassthrough()
    }) | {
        "answer": prompt | _llm | StrOutputParser(),
        "context": lambda x: x["context"]
    }


def invoke_rag_chain(rag_chain, prompt: str) -> Tuple[str, List[str]]:
    """
    Invoke a RAG chain and extract unique source references.
    
    Args:
        rag_chain: Compiled RAG chain
        prompt: Question/prompt to send to the LLM
        
    Returns:
        Tuple of (response_text, unique_sources_list)
    """
    response_dict = rag_chain.invoke(prompt)
    unique_sources = set()
    for doc in response_dict['context']:
        unique_sources.add(doc.metadata.get('source', 'Onbekende bron'))
    return response_dict['answer'], sorted(list(unique_sources))


def analyze_single_area(
    area: str,
    vector_store,
    llm,
    system_template: str,
    concept_check_prompt: str,
    table_generation_prompt: str,
    token_tracker: Optional["TokenUsageTracker"] = None,
    ai_provider: str = "Interne OpenAI"
) -> Tuple[str, Dict]:
    """
    Helper function to analyze a single area. Can be run in parallel.
    
    Args:
        area: Area name to analyze
        vector_store: Chroma vector store instance
        llm: LangChain LLM instance
        system_template: System prompt template
        concept_check_prompt: Initial concept check prompt
        table_generation_prompt: Table generation prompt template with {concept_check_result} placeholder
        token_tracker: Optional TokenUsageTracker for logging token consumption
        ai_provider: AI provider selection for token tracking
        
    Returns:
        Tuple of (area_name, result_dict) where result_dict contains 'summary', 'sources', and 'raw_data'
    """
    retriever = create_filtered_retriever(vector_store, [area])
    rag_chain = get_rag_chain(retriever, llm, system_template)
    try:
        # First pass: concept check
        concept_check_result, source_docs = invoke_rag_chain(rag_chain, concept_check_prompt)
        
        # Track tokens for concept check
        if token_tracker:
            input_tokens = token_tracker.estimate_tokens(
                system_template + concept_check_prompt,
                model="gpt-5-mini"  # Use default model for estimation
            )
            output_tokens = token_tracker.estimate_tokens(concept_check_result, model="gpt-5-mini")
            token_tracker.log_token_usage(
                provider=ai_provider.lower().replace(" ", "_"),
                operation="batch_analysis_concept_check",
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        
        # Second pass: detailed analysis
        final_prompt = table_generation_prompt.format(concept_check_result=concept_check_result)
        json_response_text, unique_sources = invoke_rag_chain(rag_chain, final_prompt)
        
        # Track tokens for table generation
        if token_tracker:
            input_tokens = token_tracker.estimate_tokens(
                system_template + final_prompt,
                model="gpt-5-mini"
            )
            output_tokens = token_tracker.estimate_tokens(json_response_text, model="gpt-5-mini")
            token_tracker.log_token_usage(
                provider=ai_provider.lower().replace(" ", "_"),
                operation="batch_analysis_table_gen",
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        
        json_data = parse_json_response(json_response_text)
        if json_data:
            formatted_markdown = format_json_to_markdown(json_data)
            return area, {'summary': formatted_markdown, 'sources': unique_sources, 'raw_data': json_data}
        else:
            return area, {'summary': f"**Fout:** Geen valide JSON.\nOutput: {json_response_text}", 'sources': unique_sources, 'raw_data': None}
    except Exception as e:
        return area, {'summary': f"Fout: {e}", 'sources': [], 'raw_data': None}


def run_batch_analysis(
    vector_store, 
    llm, 
    selected_areas: List[str], 
    concept_check_prompt: str, 
    table_generation_prompt: str, 
    system_template: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    token_tracker: Optional["TokenUsageTracker"] = None,
    ai_provider: str = "Interne OpenAI"
) -> Dict[str, Dict]:
    """
    Run batch analysis on multiple areas in parallel with optional progress tracking and token logging.
    
    Args:
        vector_store: Chroma vector store instance
        llm: LangChain LLM instance
        selected_areas: List of area names to analyze
        concept_check_prompt: Initial concept check prompt
        table_generation_prompt: Table generation prompt template with {concept_check_result} placeholder
        system_template: System prompt template
        progress_callback: Optional callback function called as progress_callback(fraction: float, message: str)
                          fraction ranges from 0 to 1; message describes current progress
                          If provided, called on each area completion during parallel processing
        token_tracker: Optional TokenUsageTracker for logging token consumption
        ai_provider: AI provider selection for token tracking
        
    Returns:
        Dictionary mapping area names to analysis results
        Each result contains 'summary' (markdown), 'sources' (list), and 'raw_data' (JSON)
    """
    if not selected_areas:
        raise ValueError("Geen documenten geselecteerd.")
    
    results = {}
    total_areas = len(selected_areas)
    
    if progress_callback:
        progress_callback(0.0, "Analyse wordt voorbereid...")
    
    # Use a thread pool to run analyses in parallel.
    # The number of workers can be tuned. 5 is a reasonable start.
    # The tasks are I/O bound (waiting for API), so we can use more threads than CPU cores.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create a list of future objects
        future_to_area = {
            executor.submit(
                analyze_single_area, 
                area, vector_store, llm, system_template, concept_check_prompt, table_generation_prompt,
                token_tracker, ai_provider
            ): area 
            for area in selected_areas
        }
        
        completed_count = 0
        # Process futures as they complete
        for future in concurrent.futures.as_completed(future_to_area):
            area = future_to_area[future]
            try:
                area_name, result_data = future.result()
                results[area_name] = result_data
            except Exception as exc:
                results[area] = {'summary': f"Fout tijdens parallelle uitvoering voor {area}: {exc}", 'sources': [], 'raw_data': None}
            
            completed_count += 1
            progress_value = completed_count / total_areas
            progress_message = f"Analyse afgerond voor: **{area}** ({completed_count}/{total_areas})"
            if progress_callback:
                progress_callback(progress_value, progress_message)
    
    # Return results in the original order of selected_areas
    ordered_results = {area: results.get(area, {'summary': 'Niet verwerkt', 'sources': [], 'raw_data': None}) for area in selected_areas}
    return ordered_results

