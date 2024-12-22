import React, { useState, useEffect, useRef } from 'react'
import { Button } from "@/components/ui/button"
import { Download } from 'lucide-react'
import { jsPDF } from 'jspdf'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import showdown from 'showdown'

const API_BASE_URL = 'http://localhost:5050';

interface ArtifactProps {
  content: string;
  title: string;
  sessionId?: string;
  onActiveThreadChange: (sessionId: string) => void;
  onReportGenerationSuccess: () => void;
}

export function Artifact({ content, title, sessionId, onActiveThreadChange, onReportGenerationSuccess }: ArtifactProps) {
  const [localContent, setLocalContent] = useState(content)
  const [activeTab, setActiveTab] = useState<'raw' | 'pretty'>('raw')
  const [htmlContent, setHtmlContent] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [amendInput, setAmendInput] = useState('')
  const [isAmendPopupOpen, setIsAmendPopupOpen] = useState(false)

  const converter = new showdown.Converter({ tables: true })
  const initialLoadRef = useRef(true)

  useEffect(() => {
    setLocalContent(typeof content === 'string' ? content : JSON.stringify(content));
  }, [content]);

  useEffect(() => {
    const safeContent = typeof localContent === 'string' ? localContent : '';
    setHtmlContent(converter.makeHtml(safeContent));
  }, [localContent]);

  // send updated report content to server when user stops editing (onBlur)
  const handleBlur = async () => {
    if (!sessionId) return;
    try {
      const formData = new FormData();
      formData.append('report', localContent);
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/report_user_input`, {
        method: 'POST',
        headers: { 'Accept': 'application/json' },
        body: formData
      });

      if (!response.ok) {
        console.error('Failed to update report content on server:', response.status, await response.text());
      }
    } catch (error) {
      console.error('Error updating report content:', error);
    }
  }

  const handleContentChange = (newContent: string) => {
    setLocalContent(newContent)
    //onActiveThreadChange?.(newContent)
  }

  const handleDownloadPDF = () => {
    const doc = new jsPDF();
    doc.html(htmlContent, {
      callback: function (doc) {
        doc.save(`${title}.pdf`);
      },
      x: 10,
      y: 10
    });
  }

  const handleGenerateReport = async () => {
    if (!sessionId) return;
    
    setIsGenerating(true);
    try {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/report`, {
        method: 'POST'
      });
  
      if (!response.ok) {
        throw new Error('Failed to generate report, error code: ' + response.status);
      }
  
      const pollReportStatus = async () => {
        try {
          const statusResponse = await fetch(`${API_BASE_URL}/session/${sessionId}/report_status`);
          const statusData = await statusResponse.json();
          if (statusData.status === 'completed') {
            onReportGenerationSuccess();
            setIsGenerating(false);
          } else if (statusData.status === 'failed') {
            setIsGenerating(false);
          } else {
            setTimeout(pollReportStatus, 10000);
          }
        } catch (error) {
          console.error('Error polling report status:', error);
          setIsGenerating(false);
        }
      };
  
      pollReportStatus();
  
    } catch (error) {
      console.error('Error generating report:', error);
      setIsGenerating(false);
    }
  };

  const handleAmendSubmit = async () => {
    if (!sessionId || !amendInput) return;

    const amendInputForm = new FormData();
    amendInputForm.append('editReport', amendInput);

    try {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}/report_update`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: amendInputForm
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to update report');
      }

      addNotification('Your changes have been shared with the server.', 'success');
      setAmendInput(''); // Clear the input after submission
      setIsAmendPopupOpen(false); // Close the popup

    } catch (error) {
      // addNotification(error instanceof Error ? error.message : 'Failed to amend report', 'error');
    }
  };

  return (
    <div className="bg-[hsl(var(--muted))] rounded overflow-hidden border border-[hsl(var(--ring))] flex flex-col h-full">
      <div className="bg-[hsl(var(--muted))] border-b border-[hsl(var(--ring))] p-2 flex justify-between items-center relative">
        <div className="flex items-center space-x-4">
          {sessionId && (
            <Button
              //variant="outline"
              size="sm"
              onClick={handleGenerateReport}
              disabled={isGenerating}
              className="w-full bg-[hsl(var(--primary))] text-[hsl(var(--foreground))]  p-2 rounded hover:bg-[hsl(var(--primary-hover))] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isGenerating ? 'Generating...' : 'Generate Report'}
            </Button>
          )}
          {sessionId && (
            <Button
              //variant="outline"
              size="sm"
              onClick={() => setIsAmendPopupOpen(true)}
              disabled={true} // disabled={!sessionId} 
              className="w-full bg-[hsl(var(--primary))] text-[hsl(var(--foreground))] p-2 rounded hover:bg-[hsl(var(--primary-hover))] disabled:cursor-not-allowed disabled:opacity-50"
            >
              Change Report
            </Button>
          )}
        </div>
        <div className="absolute left-1/2 transform -translate-x-1/2">
          <h3 className="text-[hsl(var(--foreground))]  font-semibold">{title}</h3>
        </div>
        <div className="flex items-center space-x-2">
          <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as 'raw' | 'pretty')}>
            <TabsList className="bg-[hsl(var(--popover))]  border border-[hsl(var(--ring))] rounded">
              <TabsTrigger 
                value="raw" 
                className="data-[state=active]:bg-[hsl(var(--primary))] data-[state=active]:text-[hsl(var(--foreground))]  text-[hsl(var(--foreground))]"
              >
                Raw
              </TabsTrigger>
              <TabsTrigger 
                value="pretty" 
                className="data-[state=active]:bg-[hsl(var(--primary))]  data-[state=active]:text-[hsl(var(--foreground))] text-[hsl(var(--foreground))]"
              >
                Pretty
              </TabsTrigger>
            </TabsList>
          </Tabs>
          <Button 
            //variant="outline" 
            size="sm" 
            onClick={handleDownloadPDF} 
            className="w-full bg-[hsl(var(--primary))] text-[hsl(var(--foreground))] p-2 rounded hover:bg-[hsl(var(--primary-hover))] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Download className="h-4 w-4 mr-2" />
            Download PDF
          </Button>
        </div>
      </div>
      <div className="flex-grow overflow-hidden">
        <Tabs value={activeTab} className="h-full">
          <TabsContent value="raw" className="h-full">
            <textarea
              value={localContent}
              onChange={(e) => handleContentChange(e.target.value)}
              onBlur={handleBlur}
              className="w-full h-full bg-[hsl(var(--background))] text-[hsl(var(--foreground))] border-none resize-none focus:ring-2 focus:ring-purple-500 p-4 overflow-y-auto bg-[var(--background)] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]"
              placeholder="Enter your markdown here..."
            />
          </TabsContent>
          <TabsContent value="pretty" className="h-full">
            <div 
              className="prose prose-invert max-w-none h-full bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border-none resize-none focus:ring-2 focus:ring-purple-500 p-4 overflow-y-auto bg-[var(--background)] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-[hsl(var(--muted))] [&:hover]:scrollbar-thumb-[hsl(var(--muted-foreground))]"
              dangerouslySetInnerHTML={{ __html: htmlContent }}
            />
          </TabsContent>
        </Tabs>
      </div>
      {isAmendPopupOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] p-4 rounded w-[600px] max-w-none border border-[hsl(var(--ring))]">
            <h2 className="text-lg">Change Report</h2>
            <textarea
              value={amendInput}
              onChange={(e) => setAmendInput(e.target.value)}
              placeholder="Enter your required changes here..."
              className="w-full h-64 min-h-[150px] p-2 bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] border border-[hsl(var(--ring))]"
            />
            <div className="flex justify-end mt-2">
              <Button onClick={handleAmendSubmit} className="bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-hover))] rounded">Send</Button>
              <Button onClick={() => setIsAmendPopupOpen(false)} className="ml-2 bg-transparent hover:bg-[hsl(var(--primary-hover))] border border-[hsl(var(--ring))] rounded">Cancel</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}