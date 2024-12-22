import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const path = searchParams.get('path')

  if (!path) {
    return new NextResponse('Path is required', { status: 400 })
  }

  try {
    const file = fs.readFileSync(path)
    
    return new NextResponse(file, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': 'inline',
      },
    })
  } catch (error) {
    console.error('Error reading PDF:', error)
    return new NextResponse('Error reading PDF', { status: 500 })
  }
}
