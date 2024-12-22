from server.logger import get_logger
from PIL import Image
import base64
import os
from qwen_vl_utils import process_vision_info
import json
import psutil
import torch
import fitz
import gc
from server.config import default_settings, CHUNK_SIZE, UPLOAD_FOLDER
from byaldi import RAGMultiModalModel
import re

logger = get_logger(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Global references to vision-enabled model (for chunk summaries)
MODEL = None
PROCESSOR = None
DEVICE = None

# Global references to text-only model (for final report)
TEXT_MODEL = None
TOKENIZER = None
TEXT_DEVICE = None

RAG = None
RAG_DEVICE = None

def set_global_model(model, processor, device):
    """
    called from main.py once at server startup after the vision model is loaded.
    sets global references for vision-enabled Qwen model (for chunk summaries).
    """
    global MODEL, PROCESSOR, DEVICE
    MODEL = model
    PROCESSOR = processor
    DEVICE = device

def set_global_text_model(model, tokenizer, device):
    """
    called from main.py once at server startup after the text-only model is loaded.
    sets global references for text-only Qwen model (e.g. Qwen2.5-1.5B)
    for final report generation.
    """
    global TEXT_MODEL, TOKENIZER, TEXT_DEVICE
    TEXT_MODEL = model
    TOKENIZER = tokenizer
    TEXT_DEVICE = device

def set_session_rag_model(model, device):
    global RAG, RAG_DEVICE
    RAG = model
    RAG_DEVICE = device

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

async def generate_response(query, session_data, context_info=None):
    """
    Generates a response using the selected model based on the query and context from indexed documents.
    Returns a JSON response with answer and context.
    """
    try:
        logger.info(f"Generating response using model '{session_data['settings']['vlmModels']}'.")
        session_id = session_data['id']

        if RAG is None or RAG_DEVICE is None:
            device = torch.device("mps" if torch.mps.is_available() else "cpu")
            rag = RAGMultiModalModel.from_index(os.path.abspath(os.path.join('.byaldi', session_id, '.byaldi', 'document_index')), device=device)
            set_session_rag_model(rag, device)
            logger.info(f"loaded {session_data['settings']['indexerModels']} model for chat response on session {session_id}")
                
        # Load the Qwen model
        if session_data['settings']['vlmModels'] in default_settings['vlmModels']:
            try:
                # Set device to MPS if available, otherwise CPU
                logger.info(f"loaded {session_data['settings']['vlmModels']} model for chat response on device: {DEVICE}")
                
                results = RAG.search(query, k=4)
                logger.info(f"Retrieved {len(results)} results from RAG.")
                
                # Extract document and page info
                context_info = context_info if context_info else []
                images = []
                
                # Get unique PDF paths to avoid opening the same file multiple times
                pdf_paths = {}
                for result in results:
                    try:
                        original_filename = session_data['files'][int(result.doc_id)]
                        logger.info(f"Filename is {original_filename}.")
                        pdf_path = os.path.join('uploaded_documents', session_id, original_filename)
                        if os.path.exists(pdf_path):
                            pdf_paths[result.doc_id] = {
                                'path': pdf_path,
                                'pages': [result.page_num]
                            }
                    except (IndexError, ValueError):
                        logger.warning(f"Could not find original filename for doc_id {result.doc_id}")
                
                # Process each PDF and extract required pages
                for doc_id, pdf_info in pdf_paths.items():
                    pdf_path = pdf_info['path']
                    try:
                        pdf_document = fitz.open(pdf_path)
                        logger.info(f"Opened PDF {pdf_path} with {len(pdf_document)} pages")
                        
                        for page_num in pdf_info['pages']:
                            try:
                                # PDF pages are 0-indexed in PyMuPDF
                                pdf_page = pdf_document[page_num - 1]
                                logger.info(f"Processing page {page_num} from PDF")
                                
                                # Get page size
                                page_width = pdf_page.rect.width
                                page_height = pdf_page.rect.height

                                MAX_DIMENSION = int(session_data['settings']['imageSizes'])
                                zoom = min(MAX_DIMENSION / page_width, MAX_DIMENSION / page_height)
                                
                                # Convert PDF page to image with calculated zoom
                                mat = fitz.Matrix(zoom, zoom)
                                pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
                                
                                # Convert to PIL Image using RGB mode
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                logger.info(f"Successfully converted page {page_num} to image of size {img.size}")
                                
                                # Ensure dimensions are multiples of 28 and within limits
                                width, height = img.size
                                # Calculate new dimensions that are multiples of 28
                                resized_width = min((width // 28) * 28, MAX_DIMENSION)
                                resized_height = min((height // 28) * 28, MAX_DIMENSION)
                                
                                if width != resized_width or height != resized_height:
                                    img = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
                                    logger.info(f"Resized image to {img.size}")
                                
                                images.append(img)
                                context_info.append({
                                    'filename': os.path.basename(pdf_path),
                                    'fullpath': pdf_path,
                                    'page': page_num,
                                    'score': float(results[len(context_info)].score)
                                })
                                
                            except Exception as page_error:
                                logger.error(f"Error processing page {page_num}: {str(page_error)}")
                                continue
                                
                        pdf_document.close()
                        logger.info(f"Successfully processed PDF {pdf_path}")
                        
                    except Exception as e:
                        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
                        continue

                logger.info(f"Successfully processed {len(images)} images from PDFs")

                # Prepare image contents for Qwen
                image_contents = []
                for img in images[:5]:  # Limit to top 3 relevant images
                    width, height = img.size
                    image_contents.append({
                        "type": "image",
                        "image": img,
                        "resized_height": height,
                        "resized_width": width
                    })

                # Create messages with both images and text
                messages = [
                    {
                        "role": "system",
                        "content": session_data['settings']['chatSystemPrompt'],
                        "role": "user",
                        "content": image_contents + [{"type": "text", "text": query}],
                    }
                ]

                # Process through Qwen model
                text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = PROCESSOR(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(DEVICE)

                # Generate response
                generated_ids = MODEL.generate(**inputs, max_new_tokens=4000)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response_text = PROCESSOR.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                logger.info("Response generated using Qwen model.")
                
                # Format response with context
                response = {
                    'answer': response_text,
                    'context': context_info
                }
                
                logger.info(f"Sending response: {str(response)[:200]}...")
                return response
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return {"error": str(e)}
        else:
            return {"error": f"Unsupported model choice: {session_data['settings']['vlmModels']}"}
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {"error": str(e)}


async def generate_chunk_summary(session_data, pdf_name, start_page, end_page):
    """
    generate a summary for a specific range of pages directly from the pdf (5-page chunks),
    by converting those pages to images and passing them to the Qwen model along with a prompt.
    """
    try:
        logger.info(f"generating chunk summary for pages {start_page}-{end_page} in session {session_data['id']}")
        
        # ensure model is loaded
        if MODEL is None or PROCESSOR is None or DEVICE is None:
            return {"error": "model not loaded. please ensure set_global_model is called at startup."}

        pdf_path = os.path.join(BASE_DIR, UPLOAD_FOLDER, session_data['id'], pdf_name)
        if not os.path.exists(pdf_path):
            return {"error": f"pdf file not found at {pdf_path}"}

        pdf_document = fitz.open(pdf_path)
        total_pages = pdf_document.page_count
        if end_page > total_pages:
            end_page = total_pages

        # build the query prompt
        query = f"""
        please summarize the financial and analytical content found in pages {start_page}-{end_page} of the document. 
        focus on key financial metrics, trends, and critical information that would be relevant for a financial analysis report.
        """

        images = []
        for page_num in range(start_page, end_page + 1):
            try:
                pdf_page = pdf_document[page_num - 1]
                page_width = pdf_page.rect.width
                page_height = pdf_page.rect.height
                MAX_DIMENSION = int(session_data['settings']['imageSizes'])
                zoom = min(MAX_DIMENSION / page_width, MAX_DIMENSION / page_height)
                mat = fitz.Matrix(zoom, zoom)
                pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # ensure dimensions multiple of 28
                width, height = img.size
                resized_width = min((width // 28) * 28, MAX_DIMENSION)
                resized_height = min((height // 28) * 28, MAX_DIMENSION)
                if width != resized_width or height != resized_height:
                    img = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

                images.append(img)
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                continue

        pdf_document.close()

        logger.info(f"collected {len(images)} images for pages {start_page}-{end_page}")

        image_contents = []
        # limit to top 3 images for performance
        for img in images[:5]:
            w, h = img.size
            image_contents.append({
                "type": "image",
                "image": img,
                "resized_height": h,
                "resized_width": w
            })
        logger.info(f"prepared {len(image_contents)} images for the model prompt")

        messages = [
            {
                "role": "user",
                "content": image_contents + [{"type": "text", "text": query}],
            }
        ]

        text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = PROCESSOR(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(DEVICE)

        logger.info(f"generating summary for the chunk with {session_data['settings']['vlmModels']} model")
        generated_ids = MODEL.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = PROCESSOR.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        logger.info(f"chunk summary generation completed for pages {start_page}-{end_page}")
        return {
            'summary': response_text,
            'page_range': f"{start_page}-{end_page}"
        }

    except Exception as e:
        logger.error(f"Error generating chunk summary: {str(e)}")
        return {"error": str(e)}


async def generate_response_report(session_data):
    """
    Generate a comprehensive financial analysis report from multiple PDF files by 
    iteratively summarizing chunks of pages (5 at a time) using the vision model for summaries,
    then combine them into a final text prompt and use a text-only model
    to generate the final report.
    """
    try:
        logger.info(f"Starting report generation process with language model: {session_data['settings']['languageModels']}")
        
        summaries = []
        
        # Iterate through all files in the session
        for file_info in session_data['files']:
            # Check if the file is a PDF
            if not file_info.endswith('.pdf'):
                logger.warning(f"Skipping non-PDF file: {file_info}")
                continue  # Skip non-PDF files
            
            pdf_path = os.path.join(BASE_DIR, UPLOAD_FOLDER, session_data['id'], file_info)

            logger.info(f"Verifying PDF existence at path: {pdf_path}")
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found at path: {pdf_path}")
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            logger.info(f"Opening PDF to determine total page count: {pdf_path}")
            try:
                pdf_document = fitz.open(pdf_path)
                total_pages = pdf_document.page_count
                pdf_document.close()
            except Exception as e:
                logger.error(f"Failed to open PDF '{pdf_path}': {str(e)}")
                raise e

            logger.info(f"Total pages in PDF '{file_info}': {total_pages}")

            # Begin processing in chunks
            logger.info(f"Beginning iterative chunk summary process for '{file_info}'")
            for start_page in range(1, total_pages + 1, CHUNK_SIZE):
                end_page = min(start_page + CHUNK_SIZE - 1, total_pages)
                logger.info(f"Processing pages {start_page} to {end_page} of '{file_info}'")
                
                # Generate summary for the current chunk
                try:
                    chunk_summary = await generate_chunk_summary(session_data, pdf_path, start_page, end_page)
                    if 'summary' in chunk_summary:
                        logger.info(f"Successfully generated summary for pages {start_page}-{end_page} of '{file_info}'")
                        summaries.append({
                            'filename': file_info,
                            'page_range': f"{start_page}-{end_page}",
                            'content': chunk_summary['summary']
                        })
                    else:
                        logger.warning(f"No summary generated for pages {start_page}-{end_page} of '{file_info}'")
                except Exception as e:
                    logger.error(f"Error generating summary for pages {start_page}-{end_page} of '{file_info}': {str(e)}")
                    summaries.append({
                        'filename': file_info,
                        'page_range': f"{start_page}-{end_page}",
                        'content': f"Error generating summary: {str(e)}"
                    })
                
                # Garbage collection to free up memory
                gc.collect()

        if not summaries:
            logger.error("No summaries were generated successfully; cannot proceed with report generation.")
            raise ValueError("No summaries were generated successfully.")

        logger.info("All chunks processed successfully, now combining summaries into final report prompt.")
        system_prompt = session_data['settings']['reportGenerationPrompt']
        final_report_prompt = f"""{system_prompt}

**Summarized Text:**  
{json.dumps(summaries, indent=2)}
"""
        logger.info(f"Calling the text-only model '{session_data['settings']['languageModels']}' to produce the final comprehensive report.")

        # Tokenize the prompt using the text-only tokenizer
        text = TOKENIZER.apply_chat_template(
            [{"role": "user", "content": final_report_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = TOKENIZER(
            text=[text],
            return_tensors="pt"
        ).to(TEXT_DEVICE)

        # Generate response with the text-only model
        generated_ids = TEXT_MODEL.generate(**inputs, max_new_tokens=8000)

        # Decode the generated tokens to get the final report
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        final_report = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Report generated successfully using the text-only model '{session_data['settings']['languageModels']}'.")
        return final_report

    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        logger.error(error_msg)
        return error_msg

# new way of generating report
async def retrieve_financial_data(session_data, query):
    """
    This function is a 'tool' that the text model can call to retrieve 
    financial data via RAG and vision summarization.
    """
    logger.info("Starting retrieve_financial_data tool call.")
    session_id = session_data['id']
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Query: {query}")

    if RAG is None or RAG_DEVICE is None:
        logger.info("RAG model not loaded yet, loading now...")
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        rag_path = os.path.abspath(os.path.join('.byaldi', session_id, '.byaldi', 'document_index'))
        logger.info(f"Loading RAG index from: {rag_path}")
        rag = RAGMultiModalModel.from_index(rag_path, device=device)
        set_session_rag_model(rag, device)
        logger.info(f"Loaded {session_data['settings']['indexerModels']} model for RAG retrieval on session {session_id}")
    else:
        logger.info("RAG model already loaded.")

    # Search the RAG index
    logger.info("Searching RAG index...")
    results = RAG.search(query, k=4)
    logger.info(f"Retrieved {len(results)} results from RAG.")

    context_info = []
    images = []
    pdf_paths = {}
    logger.info("Mapping RAG results to PDF pages...")
    # Identify PDF pages
    for result in results:
        try:
            original_filename = session_data['files'][int(result.doc_id)]
            pdf_path = os.path.join('uploaded_documents', session_id, original_filename)
            if os.path.exists(pdf_path):
                if result.doc_id not in pdf_paths:
                    pdf_paths[result.doc_id] = {
                        'path': pdf_path,
                        'pages': []
                    }
                pdf_paths[result.doc_id]['pages'].append(result.page_num)
                logger.info(f"Found PDF path {pdf_path} for doc_id {result.doc_id}, page {result.page_num}")
            else:
                logger.warning(f"PDF path does not exist: {pdf_path}")
        except (IndexError, ValueError):
            logger.warning(f"Could not find original filename for doc_id {result.doc_id}")

    logger.info("Converting PDF pages to images...")
    # Process PDFs and pages into images
    for doc_id, pdf_info in pdf_paths.items():
        pdf_path = pdf_info['path']
        try:
            pdf_document = fitz.open(pdf_path)
            logger.info(f"Opened PDF {pdf_path} with {len(pdf_document)} pages.")
            for page_num in pdf_info['pages']:
                try:
                    logger.info(f"Processing page {page_num} in {pdf_path}")
                    pdf_page = pdf_document[page_num - 1]
                    page_width = pdf_page.rect.width
                    page_height = pdf_page.rect.height
                    MAX_DIMENSION = int(session_data['settings']['imageSizes'])
                    zoom = min(MAX_DIMENSION / page_width, MAX_DIMENSION / page_height)
                    mat = fitz.Matrix(zoom, zoom)
                    pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    # Ensure dimensions multiple of 28
                    width, height = img.size
                    resized_width = min((width // 28) * 28, MAX_DIMENSION)
                    resized_height = min((height // 28) * 28, MAX_DIMENSION)
                    if width != resized_width or height != resized_height:
                        logger.info(f"Resizing image from {img.size} to {(resized_width, resized_height)}")
                        img = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

                    images.append(img)
                    context_info.append({
                        'filename': os.path.basename(pdf_path),
                        'fullpath': pdf_path,
                        'page': 0 if page_num == 0 else page_num-1,
                        'score': float(results[len(context_info)].score)
                    })
                    logger.info(f"Added image for page {page_num} of {pdf_path}")
                except Exception as page_error:
                    logger.error(f"Error processing page {page_num}: {str(page_error)}")
            pdf_document.close()
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")

    logger.info(f"Total images extracted: {len(images)}")
    # Now summarize these images using the vision model
    image_contents = []
    for img in images[:5]:
        w, h = img.size
        image_contents.append({
            "type": "image",
            "image": img,
            "resized_height": h,
            "resized_width": w
        })
    logger.info(f"Including top {len(image_contents)} images for summarization.")

    summarization_prompt = f"""
    Please summarize the financial and analytical content found in these retrieved pages. 
    Focus on key financial metrics, trends, and critical information relevant for a financial analysis report.
    """
    messages = [
        {
            "role": "user",
            "content": image_contents + [{"type": "text", "text": summarization_prompt}],
        }
    ]

    logger.info("Applying chat template and processing vision input...")
    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)
    logger.info("Sending prompt to vision-enabled model for summarization...")
    generated_ids = MODEL.generate(**inputs, max_new_tokens=2000)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response_text = PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    logger.info("Received summary from vision model.")
    return response_text

def try_parse_tool_calls(content: str):
    """
    Try to parse the tool calls from the model's output.
    """
    logger.debug("Parsing tool calls from model output...")
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            if isinstance(func.get("arguments"), str):
                func["arguments"] = json.loads(func["arguments"])
            tool_calls.append({"type": "function", "function": func})
            logger.debug(f"Parsed tool call: {func}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {m.group(1)} - {e}")
    if tool_calls:
        c = content[:offset].strip() if offset > 0 else ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}

async def generate_final_report(session_data):
    """
    Allows the model to call the `retrieve_financial_data` tool multiple times.
    It accumulates the results from all calls and once the model stops calling tools,
    it uses all collected data to produce a final comprehensive report.
    """

    logger.info("Starting generate_final_report process...")
    system_prompt = """Your task is to write a financial analysis report. 
    high quality and detailed information from a PDF is loaded into a database which you can query via the tool calling option you have. 
    In order to write your report, you need to call the tool "retrieve_financial_data" multiple times. 
    You need to ask specific questions for best result. Once you have enough information you can then say you have enough information to write your report.
    your report should include the following sections:
    1. executive summary:  
    - provide a concise overview of the major findings, critical insights, and their implications.  
    - summarize the overall financial health, significant drivers, and key takeaways without going into excessive detail.

    2. key financial metrics (tabular format):  
    - present a table showcasing essential financial metrics, this depends on the data provided.  
    - reference relevant summaries for deeper context.  
    - ensure clarity and easy comparison of metrics across periods.

    3. detailed analysis by section:  
    - break down the financial data and commentary from each section summary.  
    - explain the context, interpret the figures, and highlight their relevance.  
    - include hyperlinks to specific summaries where these data points originated.

    4. trends and patterns:  
    - identify notable trends, patterns, and anomalies within the financial data.  
    - discuss possible underlying causes, their potential impact on future performance, and align these patterns with insights from the summaries.

    5. conclusions:  
    - synthesize the key findings from all sections.
    """

    messages = [
        {"role": "user", "content": system_prompt}
    ]

    all_tool_results = []

    async def retrieve_financial_data_tool(query: str) -> str:
        logger.info(f"Tool 'retrieve_financial_data' called with query: {query}")
        return await retrieve_financial_data(session_data, query)


    tool_functions = {
        "retrieve_financial_data": retrieve_financial_data_tool
    }

    tools = [
        {
            "name": "retrieve_financial_data",
            "description": "This tool allows the model to read financial data from a PDF and summarize it. You need to ask specific questions for best result.",
            "arguments_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]

    def run_model_once(messages, tools):
        logger.info("Preparing prompt for text-only model...")
        # The tools passed to apply_chat_template should not include the actual Python function objects
        text = TOKENIZER.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        logger.debug(f"Prompt to model:\n{text}")
        inputs = TOKENIZER(text, return_tensors="pt").to(TEXT_DEVICE)
        logger.info("Generating response from text model...")
        outputs = TEXT_MODEL.generate(**inputs, max_new_tokens=500)
        output_text = TOKENIZER.batch_decode(outputs)[0][len(text):]
        logger.debug(f"Raw model output:\n{output_text}")
        parsed = try_parse_tool_calls(output_text)
        logger.info("Model response parsed.")
        return parsed

    max_iterations = 20
    iteration_count = 0
    logger.info("Beginning iterative loop for tool calls...")
    while iteration_count < max_iterations:
        iteration_count += 1
        logger.info(f"Iteration {iteration_count} of model responses.")
        parsed_assistant = run_model_once(messages, tools)
        messages.append(parsed_assistant)

        if "tool_calls" not in parsed_assistant:
            logger.info("No more tool calls found. Breaking loop.")
            break

        # Process each tool call
        for tool_call in parsed_assistant["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"]["arguments"]
            logger.info(f"Processing tool call: {fn_name} with args {fn_args}")

            # Retrieve the function from tool_functions mapping
            if fn_name in tool_functions:
                tool_result = await tool_functions[fn_name](fn_args["query"])
                all_tool_results.append(tool_result)
                messages.append({
                    "role": "tool",
                    "name": fn_name,
                    "content": json.dumps({"summary": tool_result})
                })
                logger.info("Tool call completed and result appended to messages.")
            else:
                logger.warning(f"No tool function found for {fn_name}, skipping.")

        gc.collect()

    logger.info("Exited tool call loop.")
    if all_tool_results:
        logger.info("Combining all tool results into a final prompt.")
        combined_summaries = "\n\n".join(all_tool_results)
        final_prompt = f"""
        Based on all the retrieved financial summaries below, please produce a comprehensive financial analysis report:
        
        {combined_summaries}
        
        The report should incorporate all key financial metrics, trends, and critical insights gathered.
        """
        messages.append({"role": "user", "content": final_prompt})
        logger.info("Requesting final report from the model...")
        final_response = run_model_once(messages, tools)
        logger.info("Final report generated.")
        return final_response["content"]
    else:
        logger.info("No tool results were collected. Returning last assistant response.")
        return parsed_assistant["content"]