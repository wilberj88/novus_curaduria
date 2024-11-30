import streamlit as st
from rag_system import extract_text_from_pdf, create_vector_store, query_vector_store, generate_response

st.title("PDF-based RAG System")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

# Process PDF
if st.button("Process PDF"):
    pdf_text = extract_text_from_pdf("uploaded_pdf.pdf")
    status = create_vector_store(pdf_text, doc_id="pdf1")
    st.write(status)

# User Query
query = st.text_input("Ask a question about the PDF:")
if query:
    # Query vector store and generate answer
    context_docs = query_vector_store(query)
    context = " ".join(context_docs)
    answer = generate_response(context, query)
    
    st.subheader("Answer:")
    st.write(answer)
