import React, { useState } from 'react'
import Image from 'next/image'
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"

interface DocumentFile {
  id: string
  name: string
  path: string
  size: number
  type: string
  uploaded_at: string
  session_id: string
}

interface DocumentGridProps {
  files: DocumentFile[]
  onFileClick: (file: DocumentFile) => void
}

export function DocumentGrid({ files, onFileClick }: DocumentGridProps) {
  const [selectedFile, setSelectedFile] = useState<DocumentFile | null>(null);

  const handleFileClick = (file: DocumentFile) => {
    setSelectedFile(file);
    if (onFileClick) {
      onFileClick(file);
    }
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Byte';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${Math.round(bytes / Math.pow(1024, i))} ${sizes[i]}`;
  };

  return (
    <div className="w-full">
      <table className="w-full">
        <thead className="bg-[hsl(var(--popover))]  border-b border-[hsl(var(--ring))]">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-medium text-[hsl(var(--foreground))] uppercase max-w-[200px]">
              Name
            </th>
            <th className="px-4 py-2 text-left text-xs font-medium text-[hsl(var(--foreground))] uppercase w-20">
              Size
            </th>
          </tr>
        </thead>
        <tbody className="bg-[hsl(var(--muted))]">
          {Array.isArray(files) && files.map((file) => (
            <tr
              key={`${file.session_id}-${file.name}`}
              className="border-t border-[hsl(var(--ring))] cursor-pointer hover:bg-[hsl(var(--popover))] transition-colors"
              onClick={() => handleFileClick(file)}
            >
              <td className="px-4 py-2 text-sm text-[hsl(var(--foreground))] truncate max-w-[200px]">
                {file.name}
              </td>
              <td className="px-4 py-2 text-sm text-[hsl(var(--foreground))]">
                {formatFileSize(file.size)}
              </td>
            </tr>
          ))}
          {(!Array.isArray(files) || files.length === 0) && (
            <tr>
              <td colSpan={2} className="px-4 py-8 text-center text-[hsl(var(--foreground))]">
                No files available
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
