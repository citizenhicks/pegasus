UPLOAD_FOLDER = 'uploaded_documents'
SESSION_FOLDER = 'sessions'
CHUNK_SIZE = 5

#default settings for frontend
default_settings = {
    'indexerModels': ['vidore/colqwen2-v1.0', 'vidore/colpali-v1.3-merged'],
    'languageModels': ['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct'],
    'vlmModels': ['Qwen/Qwen2-VL-2B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct'],
    'chatSystemPrompt': 'You have access to the uploaded documents and can answer questions about them. If you do not know the answer, just say that you don\'t know, don\'t try to make up an answer. If the question is not related to the uploaded documents, say "I can\'t help you with that."',
    'reportGenerationPrompt': """create a highly detailed and well-structured financial analysis report using the section summaries provided. the final report should be logically organized, maintain a consistent tone and style, and be formatted in markdown. whenever referencing specific sections or data points from the summaries, include hyperlinks that direct the reader back to the relevant summary. ensure that the document reads cohesively as a standalone report.

        the report should include the following sections:

        1. **executive summary**:  
        - provide a concise overview of the major findings, critical insights, and their implications.  
        - summarize the overall financial health, significant drivers, and key takeaways without going into excessive detail.

        2. **key financial metrics (tabular format)**:  
        - present a table showcasing essential financial metrics, this depends on the data provided.  
        - reference relevant summaries for deeper context.  
        - ensure clarity and easy comparison of metrics across periods.

        3. **detailed analysis by section**:  
        - break down the financial data and commentary from each section summary.  
        - explain the context, interpret the figures, and highlight their relevance.  
        - include hyperlinks to specific summaries where these data points originated.

        4. **trends and patterns**:  
        - identify notable trends, patterns, and anomalies within the financial data.  
        - discuss possible underlying causes, their potential impact on future performance, and align these patterns with insights from the summaries.

        5. **conclusions**:  
        - synthesize the key findings from all sections.  
        - provide actionable recommendations, strategic considerations, or implications for stakeholders based on the observed financial performance and identified trends.  
        - highlight any areas of uncertainty and suggest further analysis if necessary.""",
    'imageSizes': ['420','560','840','900'],
    'experimental': False
}