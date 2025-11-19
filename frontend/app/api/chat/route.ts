import { NextRequest, NextResponse } from 'next/server'
import { PDFDocument } from 'pdf-lib'
import OpenAI from 'openai'
import { generateMemoPDF, MemoData } from '@/lib/memo-generator'
import { generateMemoDocx } from '@/lib/memo-generator-docx'

// Check for API key
const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey || apiKey === 'your_openai_api_key_here') {
  console.warn('OpenAI API key not configured');
}

const openai = new OpenAI({
  apiKey: apiKey || 'sk-dummy-key' // Use dummy key to prevent initialization errors
})

// Dynamic import for pdf-parse to avoid build issues
async function extractPDFText(buffer: Buffer): Promise<{ text: string; numpages: number; info: any }> {
  try {
    const pdfParse = require('../../../lib/pdf-parse-wrapper.js');
    const data = await pdfParse(buffer);
    return data;
  } catch (error) {
    console.error('PDF parse error:', error);
    throw error;
  }
}


// Helper function to fetch project context - use direct Supabase query instead of fetch
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabaseServer = createClient(supabaseUrl, supabaseServiceKey);

async function getProjectContext(): Promise<string> {
  try {
    console.log('[getProjectContext] Fetching projects directly from Supabase...');

    // Query Supabase directly instead of making HTTP request
    const { data: projects, error, count } = await supabaseServer
      .from('projects')
      .select('*', { count: 'exact' })
      .order('npv', { ascending: false, nullsFirst: false })
      .limit(200); // Get up to 200 projects

    if (error) {
      console.error('[getProjectContext] Supabase error:', error);
      return '';
    }

    console.log('[getProjectContext] Fetched', projects?.length, 'projects');

    if (!projects || projects.length === 0) {
      console.log('[getProjectContext] No projects found');
      return '';
    }

    // Prepare data object similar to API response
    const data = {
      projects: projects.map(p => ({
        id: p.id,
        name: p.name,
        company_id: p.company_id,
        stage: p.stage || 'N/A',
        location: p.location || 'N/A',
        commodities: Array.isArray(p.commodities) ? p.commodities.join(', ') : 'N/A',
        status: p.status || 'N/A',
        npv: p.npv !== null && p.npv !== undefined ? `$${p.npv}M` : 'N/A',
        irr: p.irr !== null && p.irr !== undefined ? `${p.irr}%` : 'N/A',
        capex: p.capex !== null && p.capex !== undefined ? `$${p.capex}M` : 'N/A',
        description: p.description
      })),
      stats: {
        totalProjects: count || projects.length,
        avgIRR: projects.filter(p => p.irr !== null).reduce((sum, p) => sum + (p.irr || 0), 0) / projects.filter(p => p.irr !== null).length || 0,
        totalNPV: projects.reduce((sum, p) => sum + (p.npv || 0), 0),
        byStage: projects.reduce((acc: any, p) => {
          const stage = p.stage || 'Unknown';
          acc[stage] = (acc[stage] || 0) + 1;
          return acc;
        }, {}),
        byCommodity: projects.reduce((acc: any, p) => {
          const commodities = p.commodities || [];
          commodities.forEach((commodity: string) => {
            acc[commodity] = (acc[commodity] || 0) + 1;
          });
          return acc;
        }, {})
      }
    };
    
    if (!data.projects || data.projects.length === 0) {
      return '';
    }
    
    // Create a concise summary of projects
    let context = `\n\n### Current Mining Projects Database (${data.stats.totalProjects} projects):\n\n`;
    
    // Add summary stats
    context += `**Overview:**\n`;
    context += `- Total Projects: ${data.stats.totalProjects}\n`;
    context += `- Average IRR: ${data.stats.avgIRR?.toFixed(1)}%\n`;
    context += `- Total NPV: $${(data.stats.totalNPV / 1000).toFixed(1)}B\n\n`;
    
    // Add stage distribution
    context += `**By Stage:**\n`;
    Object.entries(data.stats.byStage || {}).forEach(([stage, count]) => {
      context += `- ${stage}: ${count}\n`;
    });
    context += '\n';
    
    // Add commodity distribution
    context += `**By Commodity:**\n`;
    Object.entries(data.stats.byCommodity || {}).forEach(([commodity, count]) => {
      context += `- ${commodity}: ${count}\n`;
    });
    context += '\n';
    
    // Add top projects by NPV
    const topProjects = data.projects
      .filter((p: any) => p.npv !== 'N/A')
      .sort((a: any, b: any) => {
        const npvA = parseFloat(a.npv.replace('$', '').replace('M', ''));
        const npvB = parseFloat(b.npv.replace('$', '').replace('M', ''));
        return npvB - npvA;
      })
      .slice(0, 10);

    if (topProjects.length > 0) {
      context += `**Top 10 Projects by NPV:**\n`;
      topProjects.forEach((p: any, i: number) => {
        context += `${i + 1}. ${p.name}: NPV ${p.npv}, IRR ${p.irr}, CAPEX ${p.capex}, Location: ${p.location}\n`;
      });
      context += '\n';
    }

    // Add comprehensive project catalog with all financial metrics
    context += `**Complete Project Catalog (${data.projects.length} projects):**\n`;
    context += 'When asked about specific projects, search this catalog for exact project names and their financial metrics.\n\n';

    data.projects.forEach((p: any) => {
      const metrics = [];
      if (p.npv !== 'N/A') metrics.push(`NPV: ${p.npv}`);
      if (p.irr !== 'N/A') metrics.push(`IRR: ${p.irr}`);
      if (p.capex !== 'N/A') metrics.push(`CAPEX: ${p.capex}`);

      context += `• ${p.name} | ${p.location} | ${p.stage} | ${p.commodities}`;
      if (metrics.length > 0) {
        context += ` | ${metrics.join(', ')}`;
      }
      context += '\n';
    });

    return context;
  } catch (error) {
    console.error('Error getting project context:', error);
    return '';
  }
}

export async function POST(req: Request) {
  console.log("Chat API called");

  // Check if API key is configured
  if (!apiKey || apiKey === 'your_openai_api_key_here' || apiKey === 'sk-dummy-key') {
    console.error("OpenAI API key not configured");
    return NextResponse.json({
      id: Date.now().toString(),
      role: "assistant",
      content: "⚠️ **Chat service is not configured**\n\nThe OpenAI API key is missing. To enable the chat functionality:\n\n1. Create a `.env.local` file in the project root\n2. Add your OpenAI API key: `OPENAI_API_KEY=your_actual_api_key`\n3. Restart the development server\n\nYou can get an API key from [OpenAI Platform](https://platform.openai.com/api-keys).",
      createdAt: new Date()
    });
  }

  try {
    const body = await req.json();
    console.log("Request type:", body.tool ? body.tool : "chat", "Request body:", JSON.stringify(body).substring(0, 200) + "...");
    
    const { messages, tool, webSearch, fileContents, databaseContext, generateMemo } = body;
    
    // Handle image generation request
    if (tool === 'image') {
      try {
        console.log("Processing image generation request");
        const lastMessage = messages[messages.length - 1].content;
        
        // Create a mining-focused prompt
        const miningPrompt = `Create a professional mining industry visualization: ${lastMessage}. 
        The image should be technical, accurate, and suitable for mining industry presentations.
        Focus on realistic depictions of mining operations, geological features, equipment, or data visualizations.`;
        
        const imageResponse = await openai.images.generate({
          model: "dall-e-3",
          prompt: miningPrompt,
          n: 1,
          size: "1024x1024",
        });
        
        const imageUrl = imageResponse.data?.[0]?.url;
        
        if (!imageUrl) {
          throw new Error("Failed to generate image: No URL returned");
        }
        
        return NextResponse.json({
          id: Date.now().toString(),
          role: "assistant",
          content: `![Generated image](${imageUrl})`,
          createdAt: new Date()
        });
      } catch (error) {
        console.error("Image generation error:", error);
        return NextResponse.json({
          id: Date.now().toString(),
          role: "assistant",
          content: "I'm sorry, I couldn't generate that image. Please try a different request.",
          createdAt: new Date()
        });
      }
    }
    
    // Add system message if it doesn't exist
    const systemMessageExists = messages.some((m: { role: string }) => m.role === 'system');
    const finalMessages = [...messages];
    
    if (!systemMessageExists) {
      finalMessages.unshift({
        role: 'system',
        content: `You are Lithos AI, an expert mining industry assistant with real-time capabilities. You specialize in:

- Mining project analysis and discovery
- Commodity market trends and pricing
- Technical mining reports and feasibility studies (NI 43-101, JORC, etc.)
- Environmental and ESG considerations in mining
- Geological and resource estimation
- Mining finance and investment analysis
- PDF document analysis and summarization
- Real-time access to mining project database

You have access to a comprehensive database of mining projects with detailed metrics including NPV, IRR, CAPEX, production rates, and ESG scores. When analyzing documents, you extract key information such as project details, resource estimates, financial metrics, and technical specifications.

Always provide data-driven insights and cite specific details from uploaded documents and the project database when available. Focus on accuracy and technical precision while remaining accessible.`
      });
    }
    
    // Add file content to the messages if provided
    if (fileContents && fileContents.length > 0) {
      console.log(`Processing ${fileContents.length} files`);
      
      let fileAnalysisPrompt = 'The user has uploaded the following files:\n\n';
      
      for (const file of fileContents) {
        if (!file) continue;
        
        console.log("Processing file:", file.fileName, "type:", file.fileType);
        
        if (file.fileType && file.fileType.startsWith('image/')) {
          // For image files, we'll add a description since gpt-5-nano doesn't have vision
          finalMessages.push({
            role: 'system',
            content: `Image file uploaded: ${file.fileName}. Note: Image analysis is not available with the current model. Please ask the user to describe the image content or provide text-based information instead.`
          });
        } else if (file.fileType === 'application/pdf') {
          // Extract text from PDF
          try {
            console.log(`Processing PDF: ${file.fileName}`);
            
            // Remove the data:application/pdf;base64, prefix
            const base64Data = file.fileContent.split(',')[1];
            const pdfBuffer = Buffer.from(base64Data, 'base64');
            
            // Try to extract text using pdf-parse
            try {
              const pdfData = await extractPDFText(pdfBuffer);
              const pageCount = pdfData.numpages;
              const extractedText = pdfData.text;
              
              console.log(`PDF has ${pageCount} pages and ${extractedText.length} characters`);
              console.log(`First 200 chars of extracted text: ${extractedText.substring(0, 200)}`);
              
              fileAnalysisPrompt += `\n### PDF Document: "${file.fileName}" (NI 43-101 Technical Report)\n\n`;
              fileAnalysisPrompt += `**Document Info:**\n`;
              fileAnalysisPrompt += `- Pages: ${pageCount}\n`;
              fileAnalysisPrompt += `- Title: ${pdfData.info?.Title || file.fileName}\n`;
              fileAnalysisPrompt += `- Author: ${pdfData.info?.Author || 'N/A'}\n\n`;
              
              // Handle text based on length
              if (extractedText.length > 100000) {
                // For very large documents, take key sections
                console.log('Large PDF detected, extracting key sections...');
                
                // Extract first 20k chars (usually contains executive summary)
                const beginning = extractedText.substring(0, 20000);
                
                // Try to find and extract key sections
                const sections = {
                  'executive summary': extractedText.match(/executive\s+summary[\s\S]{0,10000}/i)?.[0] || '',
                  'mineral resource': extractedText.match(/mineral\s+resource[\s\S]{0,10000}/i)?.[0] || '',
                  'economic analysis': extractedText.match(/economic\s+analysis[\s\S]{0,10000}/i)?.[0] || '',
                  'conclusions': extractedText.match(/conclusions?[\s\S]{0,5000}/i)?.[0] || ''
                };
                
                fileAnalysisPrompt += `**Document Content** (Key sections from ${extractedText.length} total characters):\n\n`;
                fileAnalysisPrompt += `**Beginning of Document:**\n${beginning}\n\n`;
                
                Object.entries(sections).forEach(([section, content]) => {
                  if (content) {
                    fileAnalysisPrompt += `**${section.toUpperCase()}:**\n${content.substring(0, 5000)}\n\n`;
                  }
                });
                
                finalMessages.push({
                  role: 'system',
                  content: `Large NI 43-101 Technical Report "${file.fileName}" (${pageCount} pages) has been processed. Key sections including executive summary, mineral resources, economic analysis, and conclusions have been extracted for analysis.`
                });
              } else if (extractedText.length > 30000) {
                // For medium documents, include more content
                fileAnalysisPrompt += `**Document Content** (First 30,000 characters):\n\n${extractedText.substring(0, 30000)}\n\n... (Document continues)`;
                
                finalMessages.push({
                  role: 'system',
                  content: `NI 43-101 Technical Report "${file.fileName}" (${pageCount} pages) has been processed. The first portion of the document is available for analysis.`
                });
              } else {
                // For smaller PDFs, include full text
                fileAnalysisPrompt += `**Full Document Content:**\n\n${extractedText}\n\n`;
                
                finalMessages.push({
                  role: 'system',
                  content: `NI 43-101 Technical Report "${file.fileName}" (${pageCount} pages) has been fully extracted and is available for comprehensive analysis.`
                });
              }
            } catch (textError) {
              console.error('Text extraction failed, falling back to metadata:', textError);
              
              // Fallback to pdf-lib for metadata
              const pdfDoc = await PDFDocument.load(pdfBuffer);
              const pageCount = pdfDoc.getPageCount();
              
              fileAnalysisPrompt += `PDF Document "${file.fileName}":\n`;
              fileAnalysisPrompt += `- Pages: ${pageCount}\n`;
              fileAnalysisPrompt += `- Type: Likely NI 43-101 Technical Report\n`;
              fileAnalysisPrompt += `Note: Text extraction encountered an issue. Please ensure the PDF is not encrypted or corrupted.\n\n`;
              
              finalMessages.push({
                role: 'system',
                content: `PDF Document: ${file.fileName} (${pageCount} pages). Text extraction failed, but this appears to be a mining technical report based on the filename.`
              });
            }
          } catch (error) {
            console.error('Error processing PDF:', error);
            fileAnalysisPrompt += `PDF File "${file.fileName}": [Error: ${error}]\n\n`;
            finalMessages.push({
              role: 'system',
              content: `PDF Document: ${file.fileName}. Unable to process the file.`
            });
          }
        } else if (file.fileType === 'application/json' || file.fileName.endsWith('.json')) {
          // Parse JSON files
          try {
            let jsonContent = file.fileContent;
            // If it's base64 encoded, decode it
            if (file.fileContent.includes('base64,')) {
              const base64Data = file.fileContent.split(',')[1];
              jsonContent = Buffer.from(base64Data, 'base64').toString('utf-8');
            }
            
            const jsonData = JSON.parse(jsonContent);
            fileAnalysisPrompt += `JSON File "${file.fileName}":\n\`\`\`json\n${JSON.stringify(jsonData, null, 2)}\n\`\`\`\n\n`;
          } catch (e) {
            fileAnalysisPrompt += `JSON File "${file.fileName}": [Error parsing JSON: ${e}]\n\n`;
          }
        } else if (file.fileType === 'text/csv' || file.fileName.endsWith('.csv') || 
                   file.fileType.includes('sheet') || file.fileName.endsWith('.xlsx')) {
          // Handle CSV and Excel files (common for mining data)
          let csvContent = file.fileContent;
          // If it's base64 encoded, decode it
          if (file.fileContent.includes('base64,')) {
            const base64Data = file.fileContent.split(',')[1];
            csvContent = Buffer.from(base64Data, 'base64').toString('utf-8');
          }
          
          if (file.fileName.endsWith('.xlsx')) {
            fileAnalysisPrompt += `Excel File "${file.fileName}": [Excel parsing will be implemented - for mining data, please export as CSV]\n\n`;
          } else {
            // Parse CSV
            const lines = csvContent.split('\n').filter((line: string) => line.trim());
            const headers = lines[0]?.split(',').map((h: string) => h.trim());
            
            fileAnalysisPrompt += `CSV File "${file.fileName}" (likely contains mining project data, assay results, or financial models):\n`;
            fileAnalysisPrompt += `Headers: ${headers?.join(', ') || 'No headers found'}\n`;
            fileAnalysisPrompt += `Total rows: ${lines.length - 1}\n`;
            fileAnalysisPrompt += `Sample data (first 10 rows):\n\`\`\`\n${lines.slice(0, 11).join('\n')}\n\`\`\`\n\n`;
          }
        } else {
          // For other text files
          let textContent = file.fileContent;
          // If it's base64 encoded, decode it
          if (file.fileContent.includes('base64,')) {
            const base64Data = file.fileContent.split(',')[1];
            textContent = Buffer.from(base64Data, 'base64').toString('utf-8');
          }
          
          fileAnalysisPrompt += `Text File "${file.fileName}":\n\`\`\`\n${textContent.substring(0, 3000)}${textContent.length > 3000 ? '\n... (truncated)' : ''}\n\`\`\`\n\n`;
        }
      }
      
      // Add the file analysis prompt if we have content
      if (fileAnalysisPrompt !== 'The user has uploaded the following files:\n\n') {
        console.log('Adding file analysis prompt to messages, length:', fileAnalysisPrompt.length);
        console.log('File analysis prompt preview:', fileAnalysisPrompt.substring(0, 500) + '...');
        finalMessages.push({
          role: 'system',
          content: fileAnalysisPrompt
        });
      }
    }
    
    // Add web search results to the messages if requested
    if (webSearch && webSearch.results) {
      console.log("Processing web search results");
      const formattedResults = webSearch.results.map((result: any, index: number) => 
        `${index + 1}. ${result.title}\n   ${result.snippet}\n   Link: ${result.link || '#'}`
      ).join('\n\n');
      
      finalMessages.push({
        role: 'system',
        content: `Here are web search results related to mining and the user's query:\n\n${formattedResults}\n\nPlease use these search results to provide a comprehensive answer with current information about mining projects, commodity prices, technical reports, or industry news.`
      });
    }
    
    // Add project database context with detailed instructions
    const projectContext = await getProjectContext();
    if (projectContext) {
      finalMessages.push({
        role: 'system',
        content: `MINING PROJECTS DATABASE CONTEXT:
${projectContext}

IMPORTANT: You have access to complete project data above including NPV, IRR, and CAPEX values for all projects listed.
When users ask about specific projects or financial metrics:
1. Search the project catalog above for the exact project name
2. Extract the financial data (NPV, IRR, CAPEX) directly from the catalog
3. Provide specific numeric values from the data

Example: If asked "what is the NPV of Mountain Pass REE?", search for "Mountain Pass" in the catalog above and report the NPV value shown.`
      });
    }
    
    // Add database context if provided
    if (databaseContext) {
      console.log("Including comprehensive database context with projects, companies, and news");
      finalMessages.push({
        role: 'system',
        content: `${databaseContext}

IMPORTANT DATABASE QUERYING INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have been provided with the COMPLETE database contents above, including:
• ALL mining projects with full details (NPV, IRR, CAPEX, commodity, country, stage, etc.)
• ALL companies with tickers, exchanges, market caps, and descriptions
• ALL recent news articles with titles, sources, dates, commodities, and sentiment

When the user asks questions about the database:
1. Search through the COMPLETE CATALOGS sections for exact matches
2. Filter, sort, and analyze the data based on user criteria
3. Provide specific, data-driven answers with exact numbers and names
4. Compare and rank projects/companies based on metrics like NPV, IRR, market cap
5. Identify trends across commodities, countries, or development stages
6. Cross-reference news with projects and companies when relevant

EXAMPLE QUERIES YOU CAN ANSWER:
• "Show me all lithium projects" → Search project catalog, filter where commodities contains "Lithium"
• "Which company has the highest market cap?" → Search companies catalog, sort by market_cap
• "List projects in Australia" → Filter projects where location contains "Australia"
• "What's the latest news about gold?" → Search news catalog, filter by commodities containing "Gold"
• "List all companies on the TSX exchange" → Filter companies by exchange="TSX"
• "Show production stage projects" → Filter projects by stage="Production"
• "Compare lithium companies by market cap" → Filter companies, search news for lithium-related updates

Always provide concrete data from the catalogs with specific names, numbers, and details.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
      });
    }
    
    console.log("Sending chat completion request with message count:", finalMessages.length);
    
    // Use GPT-4o-mini for all requests
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: finalMessages.map(m => {
        // Handle special case for images
        if (m.content && typeof m.content !== 'string') {
          return { role: m.role, content: m.content };
        }
        return { role: m.role, content: m.content };
      }),
      stream: false,
      max_tokens: 4096,
      temperature: 0.7,
    });
    
    const content = response.choices[0]?.message?.content || "No response generated";
    console.log("Chat completion response received, content length:", content.length);
    
    // Handle memo generation if requested
    if (generateMemo) {
      console.log("Generating investor memo...");
      
      try {
        // Extract project info from the response if available
        const projectMatch = content.match(/\*\*(.*?)\s+Project\*\*.*?Company:\s*(.*?)[\n\-]/);
        const npvMatch = content.match(/NPV.*?\$?([\d.]+)\s*[MB]/i);
        const irrMatch = content.match(/IRR.*?([\d.]+)%/i);
        
        const memoData: MemoData = {
          title: 'Investment Analysis Memo',
          date: new Date().toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
          }),
          projectName: projectMatch?.[1] || undefined,
          company: projectMatch?.[2] || undefined,
          npv: npvMatch ? parseFloat(npvMatch[1]) : undefined,
          irr: irrMatch ? parseFloat(irrMatch[1]) : undefined,
          content: content,
          recommendations: "Based on the analysis above, please review the detailed findings for investment decisions."
        };
        
        // Generate both PDF and DOCX
        const [pdfBytes, docxBytes] = await Promise.all([
          generateMemoPDF(memoData),
          generateMemoDocx(memoData)
        ]);
        
        const pdfBase64 = Buffer.from(pdfBytes).toString('base64');
        const docxBase64 = Buffer.from(docxBytes).toString('base64');
        
        console.log("Memos generated successfully");
        
        return NextResponse.json({
          id: Date.now().toString(),
          role: "assistant",
          content: content,
          createdAt: new Date(),
          memo: {
            pdf: pdfBase64,
            docx: docxBase64,
            filename: `investor-memo-${Date.now()}`
          }
        });
      } catch (memoError) {
        console.error("Memo generation error:", memoError);
        // Return response without memo if generation fails
      }
    }
    
    // Create the response object
    const responseBody = {
      id: Date.now().toString(),
      role: "assistant",
      content: content,
      createdAt: new Date()
    };
    
    console.log("Sending response:", JSON.stringify(responseBody).substring(0, 200) + "...");
    
    return NextResponse.json(responseBody);
  } catch (error: any) {
    console.error("Chat error:", error);
    
    // Provide more specific error messages
    let errorMessage = "I'm sorry, there was an error processing your request.";
    
    if (error?.message?.includes('API key') || error?.message?.includes('apiKey')) {
      errorMessage = "⚠️ **OpenAI API Key Issue**\n\nThe API key is either missing or invalid. Please:\n\n1. Create a `.env.local` file in the project root\n2. Add: `OPENAI_API_KEY=your_actual_api_key`\n3. Restart the development server\n\nGet your API key from [OpenAI Platform](https://platform.openai.com/api-keys).";
    } else if (error?.message?.includes('rate limit')) {
      errorMessage = "⚠️ **Rate Limit Exceeded**\n\nYou've hit the OpenAI API rate limit. Please wait a moment and try again.";
    } else if (error?.message?.includes('timeout')) {
      errorMessage = "⚠️ **Request Timeout**\n\nThe request took too long. This might be due to processing large files. Please try again with a smaller file or simpler query.";
    } else if (error?.message?.includes('quota')) {
      errorMessage = "⚠️ **API Quota Exceeded**\n\nYour OpenAI API quota has been exceeded. Please check your billing settings on the OpenAI platform.";
    } else if (error?.response?.data?.error) {
      errorMessage = `⚠️ **API Error**\n\n${error.response.data.error.message || error.response.data.error}`;
    } else {
      // Log the full error for debugging
      console.error("Full error details:", {
        message: error?.message,
        stack: error?.stack,
        response: error?.response
      });
      errorMessage = `⚠️ **Chat Service Error**\n\n${error?.message || 'An unexpected error occurred'}\n\nPlease check the console for more details.`;
    }
    
    // Return an error response
    return NextResponse.json({
      id: Date.now().toString(),
      role: "assistant", 
      content: errorMessage,
      createdAt: new Date()
    });
  }
}