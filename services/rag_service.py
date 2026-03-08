import streamlit as st
import concurrent.futures
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Local imports
from services.data_processing import parse_json_response, format_json_to_markdown

def create_filtered_retriever(vector_store, selected_areas):
    RETRIEVAL_COUNT = 15
    if not selected_areas:
        return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_COUNT})
    if len(selected_areas) == 1:
        chroma_filter = {"area_name": selected_areas[0]}
    else:
        chroma_filter = {"$or": [{"area_name": area} for area in selected_areas]}
    return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_COUNT, "filter": chroma_filter})

def get_rag_chain(_retriever, _llm, system_template):
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

def invoke_rag_chain(rag_chain, prompt):
    response_dict = rag_chain.invoke(prompt)
    unique_sources = set()
    for doc in response_dict['context']:
        unique_sources.add(doc.metadata.get('source', 'Onbekende bron'))
    return response_dict['answer'], sorted(list(unique_sources))

def analyze_single_area(area, vector_store, llm, system_template, concept_check_prompt, table_generation_prompt):
    """Helper function to analyze a single area. Can be run in parallel."""
    retriever = create_filtered_retriever(vector_store, [area])
    rag_chain = get_rag_chain(retriever, llm, system_template)
    try:
        concept_check_result, _ = invoke_rag_chain(rag_chain, concept_check_prompt)
        final_prompt = table_generation_prompt.format(concept_check_result=concept_check_result)
        json_response_text, unique_sources = invoke_rag_chain(rag_chain, final_prompt)
        json_data = parse_json_response(json_response_text)
        if json_data:
            formatted_markdown = format_json_to_markdown(json_data)
            return area, {'summary': formatted_markdown, 'sources': unique_sources, 'raw_data': json_data}
        else:
            return area, {'summary': f"**Fout:** Geen valide JSON.\nOutput: {json_response_text}", 'sources': unique_sources, 'raw_data': None}
    except Exception as e:
        return area, {'summary': f"Fout: {e}", 'sources': [], 'raw_data': None}

def run_batch_analysis(vector_store, llm, selected_areas, concept_check_prompt, table_generation_prompt, system_template):
    if not selected_areas: return "Geen documenten geselecteerd."
    
    results = {}
    progress_bar = st.progress(0, text="Analyse wordt voorbereid...")
    total_areas = len(selected_areas)
    
    # Use a thread pool to run analyses in parallel.
    # The number of workers can be tuned. 5 is a reasonable start.
    # The tasks are I/O bound (waiting for API), so we can use more threads than CPU cores.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create a list of future objects
        future_to_area = {
            executor.submit(
                analyze_single_area, 
                area, vector_store, llm, system_template, concept_check_prompt, table_generation_prompt
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
            progress_bar.progress(completed_count / total_areas, text=f"Analyse afgerond voor: **{area}** ({completed_count}/{total_areas})")

    progress_bar.empty()
    
    # Return results in the original order of selected_areas
    ordered_results = {area: results.get(area, {'summary': 'Niet verwerkt', 'sources': [], 'raw_data': None}) for area in selected_areas}
    return ordered_results
