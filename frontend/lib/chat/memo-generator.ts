import { PDFDocument, StandardFonts, rgb } from 'pdf-lib';

export interface MemoData {
  title: string;
  date: string;
  executive_summary: string;
  projects: {
    name: string;
    company: string;
    commodity: string;
    country: string;
    npv: number;
    irr: number;
    stage: string;
    highlights: string[];
  }[];
  market_analysis?: string;
  investment_thesis?: string;
  risks?: string[];
  recommendations?: string;
}

/**
 * Generate a professional investor memo PDF
 */
export async function generateInvestorMemoPDF(memoData: MemoData): Promise<Uint8Array> {
  // Create a new PDF document
  const pdfDoc = await PDFDocument.create();
  
  // Add fonts
  const helveticaBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);
  const helvetica = await pdfDoc.embedFont(StandardFonts.Helvetica);
  
  // Add a page
  let page = pdfDoc.addPage([612, 792]); // Letter size
  const { width, height } = page.getSize();
  
  let yPosition = height - 50;
  const leftMargin = 50;
  const rightMargin = width - 50;
  const lineHeight = 20;
  const paragraphSpacing = 10;
  
  // Title
  page.drawText('INVESTOR MEMORANDUM', {
    x: leftMargin,
    y: yPosition,
    size: 20,
    font: helveticaBold,
    color: rgb(0, 0, 0),
  });
  
  yPosition -= lineHeight * 1.5;
  
  // Date
  page.drawText(memoData.date, {
    x: leftMargin,
    y: yPosition,
    size: 10,
    font: helvetica,
    color: rgb(0.3, 0.3, 0.3),
  });
  
  yPosition -= lineHeight * 2;
  
  // Executive Summary
  page.drawText('EXECUTIVE SUMMARY', {
    x: leftMargin,
    y: yPosition,
    size: 14,
    font: helveticaBold,
    color: rgb(0, 0, 0),
  });
  
  yPosition -= lineHeight;
  
  // Break executive summary into lines
  const execSummaryLines = breakTextIntoLines(memoData.executive_summary, 90);
  for (const line of execSummaryLines) {
    if (yPosition < 100) {
      page = pdfDoc.addPage([612, 792]);
      yPosition = height - 50;
    }
    
    page.drawText(line, {
      x: leftMargin,
      y: yPosition,
      size: 10,
      font: helvetica,
      color: rgb(0, 0, 0),
    });
    yPosition -= lineHeight * 0.8;
  }
  
  yPosition -= paragraphSpacing;
  
  // Projects Section
  if (memoData.projects && memoData.projects.length > 0) {
    page.drawText('PROJECT ANALYSIS', {
      x: leftMargin,
      y: yPosition,
      size: 14,
      font: helveticaBold,
      color: rgb(0, 0, 0),
    });
    
    yPosition -= lineHeight;
    
    for (const project of memoData.projects) {
      if (yPosition < 200) {
        page = pdfDoc.addPage([612, 792]);
        yPosition = height - 50;
      }
      
      // Project name
      page.drawText(`${project.name} (${project.company})`, {
        x: leftMargin,
        y: yPosition,
        size: 12,
        font: helveticaBold,
        color: rgb(0, 0, 0),
      });
      
      yPosition -= lineHeight * 0.8;
      
      // Project details
      const details = [
        `Commodity: ${project.commodity} | Country: ${project.country}`,
        `NPV: $${(project.npv / 1000000).toFixed(1)}M | IRR: ${project.irr}% | Stage: ${project.stage}`
      ];
      
      for (const detail of details) {
        page.drawText(detail, {
          x: leftMargin + 10,
          y: yPosition,
          size: 10,
          font: helvetica,
          color: rgb(0.2, 0.2, 0.2),
        });
        yPosition -= lineHeight * 0.7;
      }
      
      // Highlights
      if (project.highlights && project.highlights.length > 0) {
        for (const highlight of project.highlights) {
          const highlightLines = breakTextIntoLines(`• ${highlight}`, 85);
          for (const line of highlightLines) {
            page.drawText(line, {
              x: leftMargin + 10,
              y: yPosition,
              size: 10,
              font: helvetica,
              color: rgb(0, 0, 0),
            });
            yPosition -= lineHeight * 0.7;
          }
        }
      }
      
      yPosition -= paragraphSpacing;
    }
  }
  
  // Market Analysis
  if (memoData.market_analysis) {
    if (yPosition < 150) {
      page = pdfDoc.addPage([612, 792]);
      yPosition = height - 50;
    }
    
    page.drawText('MARKET ANALYSIS', {
      x: leftMargin,
      y: yPosition,
      size: 14,
      font: helveticaBold,
      color: rgb(0, 0, 0),
    });
    
    yPosition -= lineHeight;
    
    const marketLines = breakTextIntoLines(memoData.market_analysis, 90);
    for (const line of marketLines) {
      if (yPosition < 100) {
        page = pdfDoc.addPage([612, 792]);
        yPosition = height - 50;
      }
      
      page.drawText(line, {
        x: leftMargin,
        y: yPosition,
        size: 10,
        font: helvetica,
        color: rgb(0, 0, 0),
      });
      yPosition -= lineHeight * 0.8;
    }
    
    yPosition -= paragraphSpacing;
  }
  
  // Investment Thesis
  if (memoData.investment_thesis) {
    if (yPosition < 150) {
      page = pdfDoc.addPage([612, 792]);
      yPosition = height - 50;
    }
    
    page.drawText('INVESTMENT THESIS', {
      x: leftMargin,
      y: yPosition,
      size: 14,
      font: helveticaBold,
      color: rgb(0, 0, 0),
    });
    
    yPosition -= lineHeight;
    
    const thesisLines = breakTextIntoLines(memoData.investment_thesis, 90);
    for (const line of thesisLines) {
      if (yPosition < 100) {
        page = pdfDoc.addPage([612, 792]);
        yPosition = height - 50;
      }
      
      page.drawText(line, {
        x: leftMargin,
        y: yPosition,
        size: 10,
        font: helvetica,
        color: rgb(0, 0, 0),
      });
      yPosition -= lineHeight * 0.8;
    }
    
    yPosition -= paragraphSpacing;
  }
  
  // Risks
  if (memoData.risks && memoData.risks.length > 0) {
    if (yPosition < 150) {
      page = pdfDoc.addPage([612, 792]);
      yPosition = height - 50;
    }
    
    page.drawText('RISK FACTORS', {
      x: leftMargin,
      y: yPosition,
      size: 14,
      font: helveticaBold,
      color: rgb(0, 0, 0),
    });
    
    yPosition -= lineHeight;
    
    for (const risk of memoData.risks) {
      const riskLines = breakTextIntoLines(`• ${risk}`, 85);
      for (const line of riskLines) {
        if (yPosition < 100) {
          page = pdfDoc.addPage([612, 792]);
          yPosition = height - 50;
        }
        
        page.drawText(line, {
          x: leftMargin,
          y: yPosition,
          size: 10,
          font: helvetica,
          color: rgb(0, 0, 0),
        });
        yPosition -= lineHeight * 0.8;
      }
    }
    
    yPosition -= paragraphSpacing;
  }
  
  // Recommendations
  if (memoData.recommendations) {
    if (yPosition < 150) {
      page = pdfDoc.addPage([612, 792]);
      yPosition = height - 50;
    }
    
    page.drawText('RECOMMENDATIONS', {
      x: leftMargin,
      y: yPosition,
      size: 14,
      font: helveticaBold,
      color: rgb(0, 0, 0),
    });
    
    yPosition -= lineHeight;
    
    const recLines = breakTextIntoLines(memoData.recommendations, 90);
    for (const line of recLines) {
      if (yPosition < 100) {
        page = pdfDoc.addPage([612, 792]);
        yPosition = height - 50;
      }
      
      page.drawText(line, {
        x: leftMargin,
        y: yPosition,
        size: 10,
        font: helvetica,
        color: rgb(0, 0, 0),
      });
      yPosition -= lineHeight * 0.8;
    }
  }
  
  // Footer on last page
  page.drawText('Confidential - For Internal Use Only', {
    x: leftMargin,
    y: 30,
    size: 8,
    font: helvetica,
    color: rgb(0.5, 0.5, 0.5),
  });
  
  // Save the PDF to bytes
  const pdfBytes = await pdfDoc.save();
  return pdfBytes;
}

/**
 * Helper function to break text into lines
 */
function breakTextIntoLines(text: string, maxCharsPerLine: number): string[] {
  const words = text.split(' ');
  const lines: string[] = [];
  let currentLine = '';
  
  for (const word of words) {
    if ((currentLine + ' ' + word).length > maxCharsPerLine) {
      if (currentLine) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        lines.push(word);
      }
    } else {
      currentLine = currentLine ? `${currentLine} ${word}` : word;
    }
  }
  
  if (currentLine) {
    lines.push(currentLine);
  }
  
  return lines;
}
