// src/components/PegasusApp.jsx

'use client';
import { Resizable } from "re-resizable";
import { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Settings, 
  Send, 
  File, 
  Paperclip,
  Moon,
  Sun,
} from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { 
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { 
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useDropzone } from 'react-dropzone';
import { Artifact } from "@/components/artifact";
import Image from 'next/image';
import { DocumentGrid } from '@/components/document-grid';
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger, DropdownMenuCheckboxItem } from '@/components/ui/dropdown-menu';
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/appSidebar";


const API_BASE_URL = 'http://localhost:5050';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  context?: SearchContext[];
}

interface IndexingProgress {
  status: 'processing' | 'completed' | 'failed';
  current_page: number;
  total_pages: number;
}

interface SearchContext {
  filename: string;
  fullpath: string;
  page: number;
  score: number;
}

interface SessionInfo {
  id: string;
  title: string;
  files: string[];
  created_at: string;
  messages: Message[];
  report: string;
  settings?: {
    indexerModels?: string;
    languageModels?: string;
    vlmModels?: string;
    imageSizes?: string;
    chatSystemPrompt?: string;
    reportGenerationPrompt?: string;
    experimental?: boolean;
  }
}

interface SettingsInfo {
  indexerModels: string[];
  languageModels: string[];
  vlmModels: string[];
  chatSystemPrompt: string;
  reportGenerationPrompt: string;
  imageSizes: string[];
}

interface AllSessionsResponse {
  sessions: { [key: string]: SessionInfo }, 
  settings: SettingsInfo
}

export function PegasusApp() {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [activeThread, setActiveThread] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [newThreadFile, setNewThreadFile] = useState<File | null>(null);
  const [newThreadTitle, setNewThreadTitle] = useState('');
  const [notifications, setNotifications] = useState<Array<{
    id: number;
    message: string;
    type: string;
  }>>([]);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [threadToDelete, setThreadToDelete] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [indexingProgress, setIndexingProgress] = useState<IndexingProgress | null>(null);
  const [reportContent, setReportContent] = useState('');
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [files, setFiles] = useState<any[]>([]);
  const [sessions, setSessions] = useState<{ [key: string]: SessionInfo }>({});
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatResizeBarRef = useRef<HTMLDivElement>(null);

  // State variables for editing thread title
  const [editDialogOpen, setEditDialogOpen] = useState<boolean>(false);
  const [editSessionId, setEditSessionId] = useState<string | null>(null);
  const [editSessionTitle, setEditSessionTitle] = useState<string>("");

  // Global options and settings
  const [indexerModelOptions, setIndexerModelOptions] = useState<string[]>([]);
  const [vlmOptions, setVlmOptions] = useState<string[]>([]);
  const [languageModelOptions, setLanguageModelOptions] = useState<string[]>([]);
  const [imageSizeOptions, setImageSizeOptions] = useState<string[]>([]);
  
  const [chatSystemPrompt, setChatSystemPrompt] = useState('');
  const [reportGenerationPrompt, setReportGenerationPrompt] = useState('');
  const [isExperimental, setIsExperimental] = useState(false);

  const [selectedSettings, setSelectedSettings] = useState({
    indexerModel: '',
    vlm: '',
    languageModel: '',
    imageSize: '',
    chatSystemPrompt: '',
    reportGenerationPrompt: '',
    experimental: false
  });

  const [isNewThreadDialogOpen, setIsNewThreadDialogOpen] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<{ name: string; path: string; currentPage?: number } | null>(null);

  const [isDarkMode, setIsDarkMode] = useState(true);

  useEffect(() => {
    document.body.classList.toggle('dark', isDarkMode);
  }, [isDarkMode]);

  const handleToggle = () => {
    setIsDarkMode(prev => !prev);
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  useEffect(() => {
    if (activeThread) {
      loadHistoryAndUpdate();
    }
  }, [activeThread]);

  useEffect(() => {
    refreshSessions();
  }, []);

  const handleActiveThreadChange = (threadId: string) => {
    setActiveThread(threadId);
    //addNotification(`Switched to thread: ${threadId}`, 'info');
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      if (acceptedFiles && acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        const fileExt = file.name.split('.').pop()?.toLowerCase();
        
        if (!fileExt || !['pdf', 'doc', 'docx'].includes(fileExt)) {
          addNotification('Unsupported file type. Please upload a PDF, DOC, or DOCX file.', 'error');
          return;
        }
        
        addNotification(`File selected: ${file.name}`, 'info');
        setNewThreadFile(file);
      }
    },
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    multiple: false
  });

  const addNotification = (message: string, type = 'info') => {
    const id = Date.now();
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  };

  const handleSendMessage = async (userMessage: string) => {
    try {
      if (!activeThread || !userMessage.trim()) {
        return;
      }
  
      // Add user message with unique ID
      const userMessageId = `user-${Date.now()}-${Math.random()}`;
      const newUserMessage: Message = {
        id: userMessageId,
        role: 'user',
        content: userMessage.trim()
      };
      
      // Use functional update to avoid state updates during render
      setChatMessages(prevMessages => [...prevMessages, newUserMessage]);
      setInputMessage('');
  
      const response = await fetch(
        `${API_BASE_URL}/session/${activeThread}/chat`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: userMessage
          }),
        }
      );
  
      const data = await response.json();
  
      if (!response.ok) {
        const errorMessage = typeof data.detail === 'string' 
          ? data.detail 
          : 'An unexpected error occurred';
        
        const errorMessageId = `system-${Date.now()}-${Math.random()}`;
        setChatMessages(prevMessages => [...prevMessages, {
          id: errorMessageId,
          role: 'system',
          content: errorMessage
        }]);
        addNotification(errorMessage, 'error');
        return;
      }
        
      const assistantMessageId = `assistant-${Date.now()}-${Math.random()}`;
      const assistantMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: data.answer,
        context: data.context
      };
      
      setChatMessages(prevMessages => [...prevMessages, assistantMessage]);
      scrollToBottom();
  
    } catch (error: any) {
      const errorMessageId = `system-${Date.now()}-${Math.random()}`;
      setChatMessages(prevMessages => [...prevMessages, {
        id: errorMessageId,
        role: 'system',
        content: `Error: ${error.message}`
      }]);
      addNotification(`Error sending message: ${error.message}`, 'error');
    }
  };

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  const handleContextClick = (context: SearchContext) => {
    setSelectedDocument({
      name: context.filename,
      path: context.fullpath,
      currentPage: context.page
    });
    //addNotification(`Opening document: ${context.filename} at page ${context.page}`, 'info');
    setShowPdfViewer(true);
  };

  const renderMessage = (message: Message, index: number) => {
    return (
      <div 
        key={message.id || `msg-${index}-${Date.now()}`} 
        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}
      >
        <div className={`max-w-[80%] ${
          message.role === 'system' 
            ? 'bg-red-500 text-[hsl(var(--foreground))]' 
            : message.role === 'user' 
              ? 'bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border border-[hsl(var(--border))]' 
              : 'bg-[hsl(var(--primary))] text-[hsl(var(--foreground))]'
        } rounded px-4 py-2`}>
          <p className="text-sm">{message.content}</p>
          {message.role === 'assistant' && message.context && (
            <div className="mt-2 text-xs text-gray-300">
              {message.context.map((ctx, ctxIndex) => (
                <button
                  key={`ctx-${message.id || index}-${ctx.filename}-${ctx.page}-${ctxIndex}`}
                  className="bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:underline"
                  onClick={() => handleContextClick(ctx)}
                >
                  [{ctx.filename} - Page {ctx.page}]
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };


  async function loadHistoryAndUpdate() { 
    try {
      const response = await fetch(`${API_BASE_URL}/session/${activeThread}/load_history`);
      if (!response.ok) {
        throw new Error('Session not found');
      }

      const data: SessionInfo = await response.json();
      setFiles(data.files || []);
      setChatMessages(data.messages || []);
      setReportContent(data.report || "");

      if (data && data.settings) {
        setSelectedSettings(prevSettings => ({
          ...prevSettings,
          indexerModel: data.settings.indexerModels || '',
          vlm: data.settings.vlmModels || '',
          languageModel: data.settings.languageModels || '',
          imageSize: data.settings.imageSizes || '',
          chatSystemPrompt: data.settings.chatSystemPrompt || '',
          reportGenerationPrompt: data.settings.reportGenerationPrompt || '',
          experimental: data.settings.experimental || false 
        }));

        setChatSystemPrompt(data.settings.chatSystemPrompt || '');
        setReportGenerationPrompt(data.settings.reportGenerationPrompt || '');
        setIsExperimental(data.settings.experimental || false); 
      }
    } catch (error: any) {
      addNotification(`Error loading history: ${error.message}`, 'error');
    }
  };

  const refreshSessions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/session/all`);
      if (!response.ok) {
        throw new Error('Failed to fetch sessions');
      }
      const data: AllSessionsResponse = await response.json();
      console.log('Fetched Sessions:', data.sessions); // Debugging line
      
      if (data && data.sessions) {
        setSessions(data.sessions);

        if (data.settings) {
          setIndexerModelOptions(data.settings.indexerModels || []);
          setLanguageModelOptions(data.settings.languageModels || []);
          setVlmOptions(data.settings.vlmModels || []);
          setImageSizeOptions(data.settings.imageSizes || []);

          if (data.settings.chatSystemPrompt?.length > 0) {
            setChatSystemPrompt(data.settings.chatSystemPrompt);
          }
          if (data.settings.reportGenerationPrompt?.length > 0) {
            setReportGenerationPrompt(data.settings.reportGenerationPrompt);
          }
        }

        // Only set active thread if none is selected and sessions exist
        if (!activeThread && Object.keys(data.sessions).length > 0) {
          setActiveThread(Object.keys(data.sessions)[0]);
        }
      } else {
        setSessions({});
      }
    } catch (error: any) {
      addNotification('Failed to refresh sessions', 'error');
    }
  }, [activeThread]);

  const validateFile = (file: File): { valid: boolean; error?: string } => {
    const fileExt = file.name.split('.').pop()?.toLowerCase();
    if (!fileExt || !['pdf', 'doc', 'docx'].includes(fileExt)) {
      return {
        valid: false,
        error: 'Unsupported file type. Please upload a PDF, DOC, or DOCX file.'
      };
    }
    return { valid: true };
  };

  const handleDeleteThread = async (threadId: string) => {
    //addNotification(`Deleting thread: ${threadId}`, 'info');
    try {
      const response = await fetch(`${API_BASE_URL}/session/${threadId}/delete`, {
        method: 'DELETE',
      });

      if (response.ok) {
        const newSessions = { ...sessions };
        delete newSessions[threadId];
        setSessions(newSessions);
        setDeleteDialogOpen(false);
        

        if (activeThread === threadId) {
          setChatMessages([]);
          const remainingSessions = Object.keys(newSessions);
          if (remainingSessions.length > 0) {
            setActiveThread(remainingSessions[0]);
          } else {
            setActiveThread(null);
          }
        }
        refreshSessions();
        addNotification('Session deleted successfully', 'success');
      } else {
        throw new Error('Failed to delete session');
      }
    } catch (error: any) {
      //console.error('Error deleting session:', error);
      addNotification('Error deleting session', 'error');
    }
  };

  const openNewThreadDialog = () => {
    setIsNewThreadDialogOpen(true);
    setNewThreadFile(null);
    setNewThreadTitle('');
    //addNotification('Opening new thread dialog', 'info');
  };

  const handleNewThreadConfirm = async () => {
    if (!newThreadFile || !newThreadTitle.trim()) {
      addNotification('Please select a file and enter a thread title', 'error');
      return;
    }

    const validation = validateFile(newThreadFile);
    if (!validation.valid) {
      addNotification(validation.error || 'Invalid file', 'error');
      return;
    }

    addNotification('Starting new thread creation...', 'info');
    setIsUploading(true);
    setIndexingProgress(null);

    try {
      const formData = new FormData();
      formData.append('file', newThreadFile);
      formData.append('title', newThreadTitle);
      
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
        },
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to upload file');
      }

      const data = await response.json();
      
      await refreshSessions();
      setActiveThread(data.session_id);
      //addNotification('Thread created successfully. Document indexing started.', 'success');

      monitorIndexingProgress(data.session_id);

    } catch (error: any) {
      //console.error('Upload error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload file';
      addNotification(errorMessage, 'error');
    } finally {
      setIsUploading(false);
      setNewThreadFile(null);
      setNewThreadTitle('');
      setIsNewThreadDialogOpen(false);
    }
  };

  const monitorIndexingProgress = async (sessionId: string) => {
    let attempts = 0;
    const maxAttempts = 240;
    //addNotification('Checking indexing status...', 'info');
    const checkStatus = async () => {
      if (attempts >= maxAttempts) {
        addNotification('Indexing is taking longer than expected', 'error');
        return;
      }

      try {
        const statusResponse = await fetch(`${API_BASE_URL}/session/${sessionId}/status`);
        const statusData = await statusResponse.json();

        if (statusData.status === 'completed') {
          setIndexingProgress(null);
          addNotification('Document processing completed', 'success');
          await refreshSessions();
        } else if (statusData.status === 'failed') {
          setIndexingProgress(null);
          addNotification('Document processing failed', 'error');
        } else if (statusData.status === 'processing') {
          setIndexingProgress(statusData);
          attempts++;
          setTimeout(checkStatus, 3000);
        }
      } catch (error: any) {
        //console.error('Status check error:', error);
        addNotification('Failed to check processing status', 'error');
      }
    };

    checkStatus();
  };

  const handleFileClick = (file: any) => {
    setSelectedDocument(file);
    setShowPdfViewer(true);
    //addNotification(`Opening file: ${file.name}`, 'info');
  };

  const handleAdditionalFileUpload = async (file: File) => {
    //addNotification('Attempting additional file upload...', 'info');
    try {
      setIsUploading(true);
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_BASE_URL}/session/${activeThread}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to upload file');
      }

      await refreshSessions();
      addNotification('Additional file uploaded successfully', 'success');

      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

    } catch (error: any) {
      //console.error('Error uploading file:', error);
      addNotification(error instanceof Error ? error.message : 'Failed to upload file', 'error');
    } finally {
      setIsUploading(false);
    }
  };

  const handleSaveChanges = async () => {
    if (!activeThread) return;
    //addNotification('Saving settings...', 'info');
    const settings = new FormData();
    settings.append('chatSystemPrompt', chatSystemPrompt);
    settings.append('reportGenerationPrompt', reportGenerationPrompt);
    settings.append('indexerModel', selectedSettings.indexerModel);
    settings.append('vlm', selectedSettings.vlm);
    settings.append('languageModel', selectedSettings.languageModel);
    settings.append('imageSize', selectedSettings.imageSize);
    settings.append('experimental', String(selectedSettings.experimental));

    await fetch(`${API_BASE_URL}/session/${activeThread}/settings_save`, {
      method: 'POST',
      body: settings,
    });
    setIsSettingsOpen(false);
    loadHistoryAndUpdate();
    addNotification('Settings saved successfully', 'success');
  };

  // Function to handle editing thread title
  const handleEditThread = (threadId: string, currentTitle: string) => {
    setEditSessionId(threadId);
    setEditSessionTitle(currentTitle);
    setEditDialogOpen(true);
  };

  const handleSaveEdit = async () => {
    if (editSessionId && editSessionTitle.trim()) {
      try {
        const response = await fetch(
          `http://localhost:5050/session/${editSessionId}/title`,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: editSessionTitle.trim() }),
          }
        );

        if (!response.ok) {
          throw new Error("Failed to update thread title");
        }
        addNotification("Thread renamed successfully", "success");

        refreshSessions();
        setEditDialogOpen(false);
        setEditSessionId(null);
        setEditSessionTitle("");
      } catch (error) {
        console.error("Error updating thread title:", error);
        addNotification("Error updating thread title", "error");
        // Optionally, add a notification or user feedback here
      }
    }
  };

  const handleCancelEdit = () => {
    setEditDialogOpen(false);
    setEditSessionId(null);
    setEditSessionTitle("");
  };

  const backgroundColor = 'hsl(var(--background))';

  return (
    <SidebarProvider>
      <AppSidebar 
        sessions={sessions}
        activeThread={activeThread}
        setActiveThread={handleActiveThreadChange}
        openNewThreadDialog={openNewThreadDialog}
        refreshSessions={refreshSessions}
        onEditThread={handleEditThread} // Pass the edit handler
        setThreadToDelete={setThreadToDelete}
        setDeleteDialogOpen={setDeleteDialogOpen}
      />
      <SidebarInset>
        <div className="flex h-screen" style={{ backgroundColor }}>
          {/* Notification System */}
          <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50 space-y-2 rounded p-2">
            {notifications.map(({ id, message, type }) => (
              <div
                key={id}
                className={`p-4 rounded shadow-lg ${
                  type === 'error' ? 'bg-red-600' : 
                  type === 'success' ? 'bg-green-600' : 
                  'bg-blue-600'
                }`}
              >
                {message}
              </div>
            ))}
          </div>

          {/* Processing indicator */}
          {indexingProgress && indexingProgress.status === 'processing' && (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-60 z-50">
              <div className="bg-gray-800 text-[hsl(var(--foreground))] px-12 py-8 rounded-xl shadow-2xl border border-gray-700 flex flex-col items-center space-y-6">
                <div className="relative w-24 h-24">
                  <Image
                    src="/favicon.ico"
                    alt="[Pegasus]"
                    width={96}
                    height={96}
                    className="absolute inset-0 z-10"
                  />
                  <Progress className="w-full h-full animate-spin opacity-50" />
                </div>
                <span className="text-2xl font-semibold tracking-wide">Processing document...</span>
              </div>
            </div>
          )}

          {/* Settings Dialog */}
          <Sheet open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
            <SheetContent side="right" className="w-[400px] sm:w-[540px] overflow-auto bg-[hsl(var(--background))] text-[hsl(var(--foreground))] border-l border-[hsl(var(--ring))] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]">
              <SheetHeader>
                <SheetTitle className="text-[hsl(var(--foreground))]">Settings</SheetTitle>
                <SheetDescription className="text-[hsl(var(--foreground))]">
                  Configure your application settings here.
                </SheetDescription>
              </SheetHeader>
                <div className="py-4">
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium">Indexer Model</label>
                      <div className="mt-1 text-[hsl(var(--foreground))] text-xs">Select the model used for indexing.</div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button className="w-full bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] p-2 rounded">
                            {selectedSettings.indexerModel || 'Select Indexer Model'}
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent className="w-56 bg-[hsl(var(--muted))]  text-[hsl(var(--foreground))] border border-[hsl(var(--ring))]">
                          {indexerModelOptions.map((option) => (
                            <DropdownMenuCheckboxItem
                              key={option}
                              checked={selectedSettings.indexerModel === option}
                              onCheckedChange={() => {
                                setSelectedSettings({ ...selectedSettings, indexerModel: option });
                                //addNotification(`Indexer Model selected: ${option}`, 'info');
                              }}
                            >
                              {option}
                            </DropdownMenuCheckboxItem>
                          ))}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                    <div>
                      <label className="text-sm font-medium">VLM (Image Model)</label>
                      <div className="mt-1 text-[hsl(var(--foreground))] text-xs">Select the model used for image processing.</div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button className="w-full bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] p-2 rounded">
                            {selectedSettings.vlm || 'Select VLM (Image Model)'}
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent className="w-56 bg-[hsl(var(--muted))]  text-[hsl(var(--foreground))] border border-[hsl(var(--ring))]">
                          {vlmOptions.map((option) => (
                            <DropdownMenuCheckboxItem
                              key={option}
                              checked={selectedSettings.vlm === option}
                              onCheckedChange={() => {
                                setSelectedSettings({ ...selectedSettings, vlm: option });
                                //addNotification(`VLM selected: ${option}`, 'info');
                              }}
                            >
                              {option}
                            </DropdownMenuCheckboxItem>
                          ))}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Language Model</label>
                      <div className="mt-1 text-hsl(var(--foreground)) text-xs">Select the model used for language processing.</div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button className="w-full bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] p-2 rounded">
                            {selectedSettings.languageModel || 'Select Language Model'}
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent className="w-56 bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border border-[hsl(var(--ring))]">
                          {languageModelOptions.map((option) => (
                            <DropdownMenuCheckboxItem
                              key={option}
                              checked={selectedSettings.languageModel === option}
                              onCheckedChange={() => {
                                setSelectedSettings({ ...selectedSettings, languageModel: option });
                                //addNotification(`Language Model selected: ${option}`, 'info');
                              }}
                            >
                              {option}
                            </DropdownMenuCheckboxItem>
                          ))}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Image Size</label>
                      <div className="mt-1 text-gray-400 text-xs">Select the size of the images.</div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button className="w-full bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] p-2 rounded">
                            {selectedSettings.imageSize || 'Select Image Size'}
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent className="w-56 bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border border-[hsl(var(--ring))]">
                          {imageSizeOptions.map((option) => (
                            <DropdownMenuCheckboxItem
                              key={option}
                              checked={selectedSettings.imageSize === option}
                              onCheckedChange={() => {
                                setSelectedSettings({ ...selectedSettings, imageSize: option });
                                //addNotification(`Image size selected: ${option}`, 'info');
                              }}
                            >
                              {option}
                            </DropdownMenuCheckboxItem>
                          ))}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    {/* A simple checkbox to toggle experimental */}
                    <div className="space-y-4">
                      <div className="flex flex-row items-center justify-between rounded border border-[hsl(var(--ring))] p-3">
                        <div className="space-y-0.5">
                          <label className="text-sm font-medium">Experimental Features</label>
                          <div className="text-xs text-[hsl(var(--foreground))]">Enable experimental features and functionality.</div>
                        </div>
                        <Switch
                          checked={selectedSettings.experimental}
                          onCheckedChange={(checked) => {
                            setSelectedSettings({ ...selectedSettings, experimental: checked });
                            setIsExperimental(checked);
                          }}
                          className="bg-[hsl(var(--primary))]"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="text-sm font-medium">Chat System Prompt</label>
                      <div className="mt-1 text-[hsl(var(--foreground))] text-xs">Enter the system prompt for the chat.</div>
                      <Textarea
                        value={chatSystemPrompt}
                        onChange={(e) => {
                          setChatSystemPrompt(e.target.value);
                          setSelectedSettings({ ...selectedSettings, chatSystemPrompt: e.target.value });
                        }}
                        className="mt-1 h-42 w-full resize-y p-2 border border-[hsl(var(--ring))] bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border-none rounded [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Report Generation System Prompt</label>
                      <div className="mt-1 text-[hsl(var(--foreground))] text-xs">Enter the system prompt for report generation.</div>
                      <Textarea
                        value={reportGenerationPrompt}
                        onChange={(e) => {
                          setReportGenerationPrompt(e.target.value);
                          setSelectedSettings({ ...selectedSettings, reportGenerationPrompt: e.target.value });
                        }}
                        className="mt-1 h-42 w-full resize-y p-2 border border-[hsl(var(--ring))] bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border-none rounded [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]"
                      />
                    </div>
                  </div>
                </div>
                <SheetFooter>
                  <Button onClick={handleSaveChanges} className="w-full bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] p-2 rounded">
                    Save Changes
                  </Button>
                </SheetFooter>
              </SheetContent>
            </Sheet>

            {/* Delete Confirmation Dialog */}
            <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
              <AlertDialogContent className="bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border border-[hsl(var(--ring))] rounded">
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Session</AlertDialogTitle>
                  <AlertDialogDescription className="text-[hsl(var(--foreground))]">
                    Are you sure you want to delete this session? This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel 
                    onClick={() => setDeleteDialogOpen(false)}
                    className="bg-transparent hover:bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] px-4 py-2 rounded"
                  >
                    Cancel
                  </AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => {
                      if (threadToDelete) {
                        handleDeleteThread(threadToDelete);
                      }
                      setDeleteDialogOpen(false);
                    }}
                    className="bg-red-600 hover:bg-red-700 text-[hsl(var(--foreground))] px-4 py-2 rounded"
                  >
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>

            {/* New Thread Dialog */}
            <Dialog open={isNewThreadDialogOpen} onOpenChange={setIsNewThreadDialogOpen}>
              <DialogContent className="sm:max-w-[425px] bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border border-[hsl(var(--ring))]">
                <DialogHeader>
                  <DialogTitle>Create New Thread</DialogTitle>
                  <DialogDescription>
                    Upload a PDF, DOC or DOCX document and give your thread a title.
                  </DialogDescription>
                </DialogHeader>

                <div className="grid gap-4 py-4">
                  <div {...getRootProps()} className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-[hsl(var(--ring))] hover:text-[hsl(var(--primary))] hover:bg-[hsl(var(--muted))] hover:text-[hsl(var(--foreground))] hover:border-[hsl(var(--ring))] transition-colors">
                    <input {...getInputProps()} />
                    {isDragActive ? (
                      <p>Drop the file here...</p>
                    ) : (
                      <p>Drag and drop a PDF, DOC, or DOCX file here, or click to select a file</p>
                    )}
                    {newThreadFile && (
                      <p className="mt-2">Selected file: {newThreadFile.name}</p>
                    )}
                  </div>

                  <div className="grid grid-cols-4 items-center gap-4">
                    <label htmlFor="threadTitle" className="text-right pr-2">
                      Title
                    </label>
                    <Input
                      id="threadTitle"
                      value={newThreadTitle}
                      onChange={(e) => setNewThreadTitle(e.target.value)}
                      placeholder="Enter thread title"
                      className="col-span-3 bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border border-[hsl(var(--ring))] rounded"
                    />
                  </div>
                </div>

                <Button
                  onClick={handleNewThreadConfirm}
                  disabled={!newThreadFile || !newThreadTitle.trim() || isUploading}
                  className="w-full bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] p-2 rounded border border-[hsl(var(--ring))] p-2 rounded"
                >
                  {isUploading ? 'Uploading...' : 'Create Thread'}
                </Button>
              </DialogContent>
            </Dialog>

            {/* PDF Viewer Dialog */}
            <Dialog open={showPdfViewer} onOpenChange={(open) => {
              setShowPdfViewer(open);
              if(!open) {
                //addNotification('Closed PDF viewer', 'info');
              }
            }}>
              <DialogContent className="max-w-4xl h-[90vh] p-0 bg-gray-800">
                <DialogTitle className="sr-only">
                  {selectedDocument?.name || 'Document Viewer'}
                </DialogTitle>
                <iframe
                  src={selectedDocument ? `/api/pdf?path=${selectedDocument.path}#page=${selectedDocument.currentPage || 1}` : ''}
                  className="w-full h-full"
                  title="PDF Viewer"
                  style={{ border: 'none' }}
                />
              </DialogContent>
            </Dialog>

            {/* Edit Thread Title Dialog */}
            <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
              <DialogContent className="sm:max-w-[425px] bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border border-[hsl(var(--ring))] rounded">
                <DialogHeader>
                  <DialogTitle>Edit Thread Title</DialogTitle>
                  <DialogDescription>
                    Update the title of your thread below.
                  </DialogDescription>
                </DialogHeader>

                <div className="py-4">
                  <Input
                    value={editSessionTitle}
                    onChange={(e) => setEditSessionTitle(e.target.value)}
                    placeholder="Enter new thread title"
                    className="bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border border-[hsl(var(--ring))] rounded"
                  />
                </div>

                <DialogFooter>
                  <Button
                    onClick={handleCancelEdit}
                    className="bg-transparent hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--foreground))] border border-[hsl(var(--ring))] px-4 py-2 rounded mr-2"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleSaveEdit}
                    disabled={!editSessionTitle.trim()}
                    className="bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] px-4 py-2 rounded"
                  >
                    Save
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            {/* Main content */}
            <div className="flex-1 flex flex-col">
              {/* Header */}
              <header className="bg-[hsl(var(--background))] border-b border-[hsl(var(--border))] rounded p-4 flex justify-between items-center">
                <div className="flex items-center">
                  {/* Sidebar toggle button can be uncommented if needed */}
                  <SidebarTrigger 
                    variant="outline"
                    size="icon" 
                    className="w-auto bg-[hsl(var(--background))] hover:bg-[hsl(var(--muted))] border border-[hsl(var(--border))] text-[hsl(var(--primary))] p-2 rounded" />
                  {/* <Button 
                    variant="outline" 
                    size="icon" 
                    onClick={() => {
                      setIsSidebarOpen(!isSidebarOpen);
                      //addNotification(`Sidebar toggled: ${!isSidebarOpen ? 'Opened' : 'Closed'}`, 'info');
                    }}
                    className="bg-[var(--primary)] text-[hsl(var(--foreground))] p-2 rounded hover:bg-[var(--primary-foreground)]"
                  >
                    {isSidebarOpen ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  </Button> */}
                  {activeThread && sessions[activeThread] && (
                    <span className="text-[hsl(var(--foreground))] font-semibold ml-14">
                      {sessions[activeThread].title || 'Untitled'}
                    </span>
                  )}
                </div>
                <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center space-x-4">
                  <h1 className="text-2xl font-bold text-[hsl(var(--foreground))]">[pegasus]</h1>
                </div>
                <div className="w-10"></div> {/* Spacer */}
                <div className="flex items-center space-x-2">
                  <Button variant="outline" size="icon" className="w-auto bg-[hsl(var(--background))] hover:bg-[hsl(var(--muted))] border border-[hsl(var(--border))] text-[hsl(var(--primary))] p-2 rounded" onClick={() => setIsSettingsOpen(true)}>
                    <Settings />
                  </Button>
                  <Button variant="outline" size="icon" className="w-auto bg-[hsl(var(--background))] hover:bg-[hsl(var(--muted))] border border-[hsl(var(--border))] text-[hsl(var(--primary))] p-2 rounded" onClick={handleToggle} disabled>
                    {isDarkMode ? <Sun /> : <Moon />}
                  </Button>
                </div>
              </header>

              {/* Content area */}
              <div className="flex-1 flex p-4 space-x-4 overflow-hidden">
                {/* Artifact view */}
                <div className="w-2/3 h-full">
                  {activeThread ? (
                    <Artifact
                      title="Report"
                      content={reportContent}
                      sessionId={activeThread}
                      onActiveThreadChange={handleActiveThreadChange} 
                      onReportGenerationSuccess={loadHistoryAndUpdate} 
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-[hsl(var(--muted))] rounded border border-[hsl(var(--ring))]">
                      <div className="text-center text-[hsl(var(--foreground))]">
                        <File className="w-12 h-12 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold mb-2">No Thread Selected</h3>
                        <p>Select or create a thread to get started</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Right side */}

                <div className="w-1/3 h-full flex flex-col">
                  <Resizable
                    defaultSize={{ width: "100%", height: "30%" }}

                  enable={{
                    right: false,
                    top: false,
                    bottom: true,
                    left: false,
                    topRight: false,
                    bottomRight: false,
                    bottomLeft: false,
                    topLeft: false,
                  }}
                  className="overflow-hidden rounded"
                  style={{ display: "flex", flexDirection: "row", }}

                  >
                  {/* Tabs section */}
                  <div className="bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded overflow-hidden w-full">
                    {activeThread ? (
                      <Tabs defaultValue="files" className="h-full flex flex-col">
                        <div className="bg-[hsl(var(--background))] border-b border-[hsl(var(--border))] p-2 flex justify-between items-center rounded">
                          <TabsList className="bg-[hsl(var(--popover))]  border border-[hsl(var(--border))] rounded">
                            <TabsTrigger 
                              value="files" 
                              className="data-[state=active]:bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] rounded"
                              //onClick={() => addNotification('Viewing Files tab', 'info')}
                            >
                              Files
                            </TabsTrigger>
                            <TabsTrigger 
                              value="table" 
                              className="data-[state=active]:bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] rounded"
                             // onClick={() => addNotification('Viewing Table tab', 'info')}
                            >
                              Table
                            </TabsTrigger>
                            <TabsTrigger 
                              value="chart" 
                              className="data-[state=active]:bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] rounded"
                             // onClick={() => addNotification('Viewing Chart tab', 'info')}
                            >
                              Chart
                            </TabsTrigger>
                            <TabsTrigger 
                              value="news" 
                              className="data-[state=active]:bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] text-[hsl(var(--primary-foreground))] rounded"
                             // onClick={() => addNotification('Viewing News tab', 'info')}
                            >
                              News
                            </TabsTrigger>
                          </TabsList>
                          <Button
                            size="icon"
                            variant="outline"
                            onClick={() => {
                              fileInputRef.current?.click();
                              //addNotification('Select a file to upload', 'info');
                            }}
                            className="w-auto bg-[hsl(var(--background))] hover:bg-[hsl(var(--muted))] border border-[hsl(var(--border))] text-[hsl(var(--primary))] p-2 rounded"
                          >
                            <Paperclip />
                          </Button>
                          <input
                            type="file"
                            ref={fileInputRef}
                            style={{ display: 'none' }}
                            onChange={(e) => {
                              if(e.target.files?.[0]) {
                               // addNotification(`Additional file chosen: ${e.target.files[0].name}`, 'info');
                                handleAdditionalFileUpload(e.target.files?.[0]!);
                              }
                            }}
                            accept=".pdf,.doc,.docx"
                          />
                        </div>
                        <TabsContent value="files" className="flex-1 p-4 overflow-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]">
                          <DocumentGrid files={files} onFileClick={handleFileClick} />
                        </TabsContent>
                        <TabsContent value="table" className="flex-1 p-4 overflow-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]">
                          Table content here...
                        </TabsContent>
                        <TabsContent value="chart" className="flex-1 p-4 overflow-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]">
                          Chart content here...
                        </TabsContent>
                        <TabsContent value="news" className="flex-1 p-4 overflow-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]">
                          News content here...
                        </TabsContent>
                      </Tabs>
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-[hsl(var(--muted))] rounded border border-[hsl(var(--ring))]">
                        <div className="text-center text-[hsl(var(--foreground))]">
                          <File className="w-12 h-12 mx-auto mb-4" />
                          <h3 className="text-xl font-semibold mb-2">No Thread Selected</h3>
                          <p>Select or create a thread to get started</p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  </Resizable>
                  {/* Chat */}
                  <div className="h-2"></div>
                  <div className="h-full bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded p-4 flex flex-col overflow-hidden">
                    {activeThread ? (
                      <>
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="font-bold text-[hsl(var(--foreground))]">Chat</h3>
                        </div>
                        <div 
                          ref={chatContainerRef}
                          className="group flex-1 overflow-y-auto p-4 space-y-4 chat-messages [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]"
                        >
                          {chatMessages.map((message, index) => renderMessage(message, index))}
                        </div>
                        <div className="flex items-center space-x-2">
                          <Input 
                            placeholder="Type your message..." 
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            onKeyPress={(e) => {
                              if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSendMessage(inputMessage);
                              }
                            }}
                            className="bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] placeholder-[hsl(var(--muted-foreground))] border border-[hsl(var(--border))] rounded"
                          />
                          <Button size="icon" variant="outline" onClick={() => handleSendMessage(inputMessage)} className="w-auto bg-[hsl(var(--background))] hover:bg-[hsl(var(--muted))] border border-[hsl(var(--border))] text-[hsl(var(--primary))] p-2 rounded">
                            <Send/>
                          </Button>
                        </div>
                      </>
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-[hsl(var(--muted))] rounded">
                        <div className="text-center text-[hsl(var(--foreground))]">
                          <File className="w-12 h-12 mx-auto mb-4" />
                          <h3 className="text-xl font-semibold mb-2">No Thread Selected</h3>
                          <p>Select or create a thread to get started</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
              </div>
              
            </div>
            
          </div>
          
        </SidebarInset>
        
      </SidebarProvider>
  );
}