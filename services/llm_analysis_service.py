import os
import tempfile
import streamlit as st

def analyze_local_pdf(uploaded_file, client, system_prompt, impact_prompt):
    from langchain_community.document_loaders import PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        prompt_content = impact_prompt.format(context=full_text)
        
        # Bepaal het model op basis van de provider
        model_name = "gemini-2.5-flash" if st.session_state.ai_provider == "Mijn Gemini" else "gpt-5-mini"

        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_content}
            ],
            temperature=1 
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"Fout bij analyseren Omgevingsvisie: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def generate_conclusion_paragraph(topic: str, impact_analyse: str, client, conclusion_prompt_template) -> str:
    """Genereert een enkele conclusieparagraaf voor een specifiek onderwerp op basis van impactanalyse."""
    if not client:
        return f"*(Fout: OpenAI client niet beschikbaar voor onderwerp '{topic}')*"
    
    prompt_content = conclusion_prompt_template.format(
        topic=topic,
        impact_analyse=impact_analyse
    )
    
    try:
        # Bepaal het model op basis van de provider
        model_name = "gemini-2.5-flash" if st.session_state.ai_provider == "Mijn Gemini" else "gpt-5-mini"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Je bent een expert in ecologie en ruimtelijke ordening."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=1 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"*(Fout bij het genereren van conclusie voor '{topic}': {e})*"
