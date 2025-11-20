// Edge-compatible PDF parsing wrapper
// This avoids issues with dynamic requires on Vercel

module.exports = async function parsePDF(dataBuffer, options) {
  try {
    // Use dynamic import instead of require
    const pdfParse = await import('pdf-parse/lib/pdf-parse.js');
    const pdf = pdfParse.default || pdfParse;
    
    // Default options
    options = options || {};
    const defaultOptions = {
      max: 0,
      version: 'default',
    };
    
    // Make sure the buffer is in the right format
    const data = dataBuffer.data ? dataBuffer : { data: dataBuffer };
    
    return await pdf(data, Object.assign({}, defaultOptions, options));
  } catch (error) {
    console.error('PDF parse error in edge runtime:', error);
    throw error;
  }
};