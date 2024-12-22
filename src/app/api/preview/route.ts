import { NextRequest, NextResponse } from 'next/server'
import { PDFDocument } from 'pdf-lib'
import fs from 'fs'
import path from 'path'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const filePath = searchParams.get('path')

  if (!filePath) {
    return new NextResponse('Path is required', { status: 400 })
  }

  try {
    // Decode the URL-encoded path
    const decodedPath = decodeURIComponent(filePath)
    console.log('Attempting to read file:', decodedPath)

    if (!fs.existsSync(decodedPath)) {
      console.error('File not found:', decodedPath)
      return new NextResponse('File not found', { status: 404 })
    }

    const pdfBytes = fs.readFileSync(decodedPath)
    console.log('Successfully read PDF file of size:', pdfBytes.length)

    const pdfDoc = await PDFDocument.load(pdfBytes)
    const pages = pdfDoc.getPages()
    
    if (pages.length === 0) {
      console.error('PDF has no pages')
      return new NextResponse('PDF has no pages', { status: 400 })
    }

    // Get the first page
    const firstPage = pages[0]
    const pngImage = await firstPage.render({
      width: firstPage.getWidth(),
      height: firstPage.getHeight(),
    })
    
    const pngBytes = await pngImage.save()
    console.log('Successfully generated preview of size:', pngBytes.length)
    
    return new NextResponse(pngBytes, {
      headers: {
        'Content-Type': 'image/png',
        'Cache-Control': 'public, max-age=31536000',
      },
    })
  } catch (error) {
    console.error('Error generating preview:', error)
    return new NextResponse(`Error generating preview: ${error.message}`, { status: 500 })
  }
}
