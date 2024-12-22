import os
import uuid
import json
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
from server.indexer import index_documents
from server.responder import generate_response, generate_response_report, set_global_model, set_global_text_model, set_session_rag_model, generate_final_report
from server.logger import get_logger
from server.model_loader import load_model, load_text_model
from server.config import default_settings, UPLOAD_FOLDER, SESSION_FOLDER
import threading
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager
from byaldi import RAGMultiModalModel
import torch
import asyncio

# Set the TOKENIZERS_PARALLELISM environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_FOLDER = os.path.join(BASE_DIR, '.byaldi')
processing_status = {}
report_generation_status = {}
session_locks = {}
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(BASE_DIR, UPLOAD_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, SESSION_FOLDER), exist_ok=True)
    os.makedirs(INDEX_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR,'static'), exist_ok=True)

    # RAG model for chat responses
    TEXT_MODEL, TOKENIZER, TEXT_DEVICE = load_text_model(default_settings['languageModels'][0])
    set_global_text_model(TEXT_MODEL, TOKENIZER, TEXT_DEVICE)
    logger.info(f"loaded {default_settings['languageModels'][0]} model for text-only responder on device: {TEXT_DEVICE}")

    # Vision model
    MODEL, PROCESSOR, DEVICE = load_model(default_settings['vlmModels'][0])
    set_global_model(MODEL, PROCESSOR, DEVICE)
    logger.info(f"loaded {default_settings['vlmModels'][0]} vision model for responder on device: {DEVICE}")

    initialize_session_locks()
    logger.info("Session locks initialized for existing sessions.")

    try:
        yield
    finally:
        pass

app = FastAPI(title="PegasusBackend", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialize_session_locks():
    """
    Initialize locks for all existing sessions on server startup.
    """
    sessions = get_list_of_sessions()
    for session_id in sessions:
        if session_id not in session_locks:
            session_locks[session_id] = threading.Lock()
    return

def get_list_of_sessions() -> list:
    try:
        session_folder = os.path.join(BASE_DIR, SESSION_FOLDER)
        return [file.replace('.json', '') for file in os.listdir(session_folder) if file.endswith('.json')]
    except Exception as e:
        logger.error(f"Error getting list of sessions: {str(e)}")
        return []

def get_session_data(session_id: str) -> dict:
    session_file = os.path.join(BASE_DIR, SESSION_FOLDER, f'{session_id}.json')
    if not os.path.exists(session_file):
        raise HTTPException(status_code=404, detail="Session not found")
    with open(session_file, 'r') as f:
        return json.load(f)

def save_session_data(session_id: str, data: dict):
    session_file = os.path.join(BASE_DIR, SESSION_FOLDER, f'{session_id}.json')
    temp_file = session_file + '.tmp'  # For atomic write
    with open(temp_file, 'w') as f:
        json.dump(data, f)
    os.replace(temp_file, session_file)

@app.post("/upload")
async def new_session(file: UploadFile = File(...), title: str = Form(None)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No files provided")

        session_id = str(uuid.uuid4())
        session_dir = os.path.join(BASE_DIR, UPLOAD_FOLDER, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Handle file upload
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Save the uploaded file
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        session_data = {
            'id': session_id,
            'title': title if title else file.filename,
            'files': [file.filename],
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'messages': [],
            'report': '',
            'settings': {
                'indexerModels': default_settings['indexerModels'][0],
                'languageModels': default_settings['languageModels'][0],
                'vlmModels': default_settings['vlmModels'][0],
                'chatSystemPrompt': default_settings['chatSystemPrompt'].strip(),
                'reportGenerationPrompt': default_settings['reportGenerationPrompt'].strip(),
                'imageSizes': default_settings['imageSizes'][3],
                'experimental': default_settings['experimental']
            }
        }
        
        # Initialize lock for the new session
        session_locks[session_id] = threading.Lock()
        
        # Save session data within the lock
        with session_locks[session_id]:
            save_session_data(session_id, session_data)

        # Index these files in a thread
        index_path = os.path.join(INDEX_FOLDER, session_id)
        os.makedirs(index_path, exist_ok=True)
        
        # Initialize processing status
        processing_status[session_id] = {
            'status': 'processing',
            'progress': None
        }
        
        # Start indexing in a separate thread
        def update_status(progress_info):
            if isinstance(progress_info, dict) and progress_info.get('status') == 'completed':
                processing_status[session_id]['status'] = 'completed'
            else:
                processing_status[session_id]['status'] = 'processing'

        thread = threading.Thread(
            target=index_documents,
            args=(os.path.join(session_dir, file.filename),),
            kwargs={
                'index_path': index_path,
                'indexer_model': session_data['settings']['indexerModels'],
                'progress_callback': update_status,
                'add_to_existing': False
            }
        )
        thread.start()

        return JSONResponse({
            'session_id': session_id,
            'message': 'Session created and document processing started',
            'session_info': session_data
        })
    except Exception as e:
        logger.error(f"Error in new_session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/upload")
async def add_file_to_session(session_id: str, file: UploadFile = File(...)):
    try:
        sessions_list = get_list_of_sessions()
        if session_id not in sessions_list:
            raise HTTPException(status_code=404, detail="Session not found")
        
        supported_extensions = ['.pdf', '.doc', '.docx']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported formats: {', '.join(supported_extensions)}"
            )

        session_dir = os.path.join(BASE_DIR, UPLOAD_FOLDER, session_id)
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add to existing index in a thread
        index_path = os.path.join(INDEX_FOLDER, session_id)

        # Update session info within the lock
        with session_locks[session_id]:
            session_data = get_session_data(session_id)
            session_data['files'].append(file.filename)
            save_session_data(session_id, session_data)

        # Reset processing status
        processing_status[session_id] = {'status': 'processing', 'progress': None}

        def update_status(progress_info):
            if isinstance(progress_info, dict) and progress_info.get('status') == 'completed':
                processing_status[session_id]['status'] = 'completed'
            else:
                processing_status[session_id]['status'] = 'processing'

        thread = threading.Thread(
            target=index_documents,
            args=(os.path.join(session_dir, file.filename),),
            kwargs={
                'index_path': index_path,
                'indexer_model': session_data['settings']['indexerModels'],
                'progress_callback': update_status,
                'add_to_existing': True  # <-- important
            }
        )
        thread.start()

        return JSONResponse({
            'message': 'File added to session and processing started',
            'file_info': {
                'id': str(uuid.uuid4()),
                'name': file.filename,
                'path': file_path,
                'size': os.path.getsize(file_path),
                'type': file.content_type,
                'uploaded_at': datetime.now().isoformat()
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding file to session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/status")
async def get_processing_status(session_id: str):
    sessions_list = get_list_of_sessions()
    if session_id not in sessions_list:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session is not processing")
    return processing_status[session_id]

@app.get("/session/all")
async def get_all_sessions():
    try:
        sessions_list = get_list_of_sessions()
        if not sessions_list:
            return {"sessions": {}}
            
        sessions_info = {}
        for session in sessions_list:
            try:
                session_data = get_session_data(session)
                if session_data:
                    sessions_info[session] = {
                        "id": session_data.get("id", session),
                        "title": session_data.get("title", "Untitled"),
                        "files": session_data.get("files", []),
                        "created_at": session_data.get("created_at", ""),
                        "messages": session_data.get("messages", []),
                        "report": session_data.get("report", ""),
                        "settings": session_data.get("settings", {})
                    }
            except Exception as e:
                logger.error(f"Error loading session {session}: {str(e)}")
                continue
        return {"sessions": sessions_info, "settings": default_settings}
    except Exception as e:
        logger.error(f"Error in get_all_sessions: {str(e)}")
        return {"sessions": {}}

@app.get("/session/{session_id}/load_history")
async def get_session(session_id: str, background_tasks: BackgroundTasks):
    sessions_list = get_list_of_sessions()
    if session_id not in sessions_list:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = get_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session data not found")

    files_info = []
    session_dir = os.path.join(BASE_DIR, UPLOAD_FOLDER, session_id)
    for filename in os.listdir(session_dir):
        file_path = os.path.join(session_dir, filename)
        if os.path.isfile(file_path):
            file_stat = os.stat(file_path)
            files_info.append({
                "id": f"{session_id}-{filename}",
                "name": filename,
                "path": file_path,
                "size": file_stat.st_size,
                "type": "application/pdf",
                "uploaded_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "session_id": session_id
            })

    background_tasks.add_task(load_rag_for_session, session_id, session_data)

    return {
        "id": session_data.get("id"),
        "title": session_data.get("title"),
        "files": files_info,
        "created_at": session_data.get("created_at"),
        "messages": session_data.get("messages", []),
        "report": session_data.get("report", ""),
        "settings": session_data.get("settings", {})
    }

def load_rag_for_session(session_id: str, session_data: dict):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    index_path = os.path.abspath(os.path.join('.byaldi', session_id, '.byaldi', 'document_index'))
    if os.path.exists(index_path):
        rag = RAGMultiModalModel.from_index(index_path, device=device)
        set_session_rag_model(rag, device)
        logger.info(f"loaded {session_data['settings']['indexerModels']} model for chat response on session {session_id}")
    else:
        logger.info(f"No model found for chat response on session {session_id}")

@app.post("/session/{session_id}/report")
async def generate_report(session_id: str, background_tasks: BackgroundTasks):
    sessions_list = get_list_of_sessions()
    if session_id not in sessions_list:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = get_session_data(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session data not found")
        
        if not session_data.get('files'):
            raise HTTPException(status_code=400, detail="No files found in session")
    
        # Initialize report generation status
        report_generation_status[session_id] = {"status": "processing", "error": None}
        logger.info(f"Starting background report generation for session {session_id}")
        
        # Add the background task
        background_tasks.add_task(generate_report_background, session_id)
        
        return JSONResponse(
            {"detail": "Report generation started in background.", "status": "processing"},
            status_code=200
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in report endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_report_background(session_id: str):
    try:
        session_data = get_session_data(session_id)
        if not session_data:
            raise ValueError("Session data not found")
        
        if not session_data.get('files'):
            raise ValueError("No files found in session")
    
        # Choose the appropriate report generation function
        if session_data['settings']['experimental']:
            # Assuming generate_final_report is an async function
            report = asyncio.run(generate_final_report(session_data))
        else:
            # Assuming generate_response_report is an async function
            report = asyncio.run(generate_response_report(session_data))
        
        with session_locks[session_id]:
            session_data = get_session_data(session_id)
            session_data['report'] = report
            save_session_data(session_id, session_data)
        
        # Update the report generation status
        report_generation_status[session_id] = {"status": "completed", "error": None}
        logger.info(f"Report generation completed for session {session_id}")
    except Exception as e:
        logger.error(f"Error in background report generation: {str(e)}")
        report_generation_status[session_id] = {"status": "failed", "error": str(e)}

@app.get("/session/{session_id}/report_status")
async def get_report_status(session_id: str):
    sessions_list = get_list_of_sessions()
    if session_id not in sessions_list:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_id not in report_generation_status:
        return {"status": "not_started"}
    return report_generation_status[session_id]

@app.post("/session/{session_id}/chat")
async def chat(request: dict, session_id: str):
    sessions_list = get_list_of_sessions()
    if session_id not in sessions_list:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        if session_id not in session_locks:
            # Initialize lock if it doesn't exist (safety check)
            session_locks[session_id] = threading.Lock()
        
        with session_locks[session_id]:
            session_data = get_session_data(session_id)
            if 'query' not in request:
                raise HTTPException(status_code=400, detail="Query is required")
                
            response = await generate_response(request['query'], session_data)
            session_data["messages"].append({
                "role": "user",
                "content": request['query'],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            session_data["messages"].append({
                "role": "assistant",
                "content": response.get("answer", ""),
                "context": response.get("context", []),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_session_data(session_id, session_data)
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}/delete")
async def delete_session(session_id: str):
    try:
        sessions_list = get_list_of_sessions()
        if session_id not in sessions_list:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Acquire the lock before deleting to ensure no other thread is modifying the session
        if session_id in session_locks:
            session_locks[session_id].acquire()
        
        try:
            # Load session information
            session_data = get_session_data(session_id)
            
            # Delete session file
            session_file = os.path.join(BASE_DIR, SESSION_FOLDER, f'{session_id}.json')
            if os.path.exists(session_file):
                os.remove(session_file)
            
            # Delete index folder
            index_folder = os.path.join(INDEX_FOLDER, session_id)
            if os.path.exists(index_folder):
                shutil.rmtree(index_folder)
            
            # Delete uploaded files
            upload_folder = os.path.join(BASE_DIR, UPLOAD_FOLDER, session_id)
            if os.path.exists(upload_folder):
                shutil.rmtree(upload_folder)
            
            # Remove the session lock
            if session_id in session_locks:
                del session_locks[session_id]
        
        finally:
            if session_id in session_locks:
                session_locks[session_id].release()
        
        return JSONResponse({"message": "Session deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/session/{session_id}/title")
async def update_session(session_id: str, request: dict):
    sessions_list = get_list_of_sessions()
    if session_id not in sessions_list:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        with session_locks[session_id]:
            session_data = get_session_data(session_id)
            session_data['title'] = request['title']
            save_session_data(session_id, session_data)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/report_user_input")
async def update_report(session_id: str, report: str = Form(None)):
    sessions_list = get_list_of_sessions()
    if session_id not in sessions_list:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        with session_locks[session_id]:
            session_data = get_session_data(session_id)
            session_data['report'] = report
            save_session_data(session_id, session_data)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/settings_save")
async def settings_save(
    session_id: str,
    chatSystemPrompt: str = Form(''),
    reportGenerationPrompt: str = Form(''),
    indexerModel: str = Form(''),
    vlm: str = Form(''),
    languageModel: str = Form(''),
    imageSize: str = Form(''),
    experimental: bool = Form(False)
):
    sessions_list = get_list_of_sessions()
    if session_id not in sessions_list:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        with session_locks[session_id]:
            session_data = get_session_data(session_id)
            session_data['settings'] = {
                'indexerModels': indexerModel,
                'languageModels': languageModel,
                'vlmModels': vlm,
                'chatSystemPrompt': chatSystemPrompt,
                'reportGenerationPrompt': reportGenerationPrompt,
                'imageSizes': imageSize,
                'experimental': experimental
            }
            save_session_data(session_id, session_data)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/charts")
async def charts():
    pass

@app.get("/session/{session_id}/news")
async def news():
    pass

@app.get("/session/{session_id}/tables")
async def tables():
    pass

if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5050,
        reload=True,
        reload_dirs=["models"],
        reload_excludes=[
            ".venv/*",
            "__pycache__/*",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "node_modules/*",
            "uploaded_documents/*",
            "sessions/*"
        ]
    )