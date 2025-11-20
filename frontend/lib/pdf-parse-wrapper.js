// Wrapper for pdf-parse to avoid the test file issue
const pdf = require('pdf-parse/lib/pdf-parse');

module.exports = function(dataBuffer, options) {
  // Default options
  options = options || {};
  
  // pdf-parse options
  const defaultOptions = {
    // Max number of pages to parse
    max: 0,
    // Verbosity level
    version: 'default',
  };
  
  // Make sure the buffer is in the right format
  // PDFJS expects data to be in { data: buffer } format
  const data = dataBuffer.data ? dataBuffer : { data: dataBuffer };
  
  return pdf(data, Object.assign({}, defaultOptions, options));
};