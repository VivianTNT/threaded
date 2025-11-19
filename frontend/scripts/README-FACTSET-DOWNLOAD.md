# FactSet Document Download Scripts

## Quick Start

Once your FactSet credentials are working, use these scripts to download mining documents:

### 1. Verify Credentials (Run This First)

```bash
npx tsx scripts/verify-factset-credentials.ts
```

**Expected Output** (when working):
```
✅ ALL TESTS PASSED!
Your FactSet credentials are valid and the API is accessible.
```

**Current Output**:
```
❌ ALL TESTS FAILED
403 Forbidden - User Authorization Failed
```

➡️ **Action**: Contact FactSet support if you see all failures

---

### 2. Download Sample Documents

```bash
npx tsx scripts/test-factset-download-final.ts
```

**What it does**:
- Searches for recent SEDAR/SEDAR+ filings for 3 Canadian mining companies
- Displays document metadata for each filing
- Attempts to download up to 3 documents per company
- Saves files to `downloads/factset-test/`

**Example Output** (once working):
```
📊 Ivanhoe Mines (IVN-CA)
   ✅ Found 12 filing(s)

   📄 Filing 1:
      Title: NI 43-101 Technical Report - Kamoa-Kakula Copper Project
      Date: 3/15/2024
      Form: Technical Report
      ID: DOC123456
      Size: 15.2 MB

      📥 Downloading...
      ✅ Downloaded 15234 KB (application/pdf)
      💾 Saved: IVN_1_DOC123456.pdf
```

---

### 3. Test Specific Document URL

```bash
npx tsx scripts/test-factset-simple.ts
```

Uses the specific filing URL from your email to test direct document access.

---

## Troubleshooting

### Issue: "403 Forbidden - User Authorization Failed"

**Meaning**: The API credentials aren't being accepted

**Solutions**:
1. Verify `FACTSET_USERNAME` and `FACTSET_API_KEY` in `.env.local`
2. Contact FactSet support to verify account is provisioned for API access
3. Ask if IP whitelisting is required
4. Confirm the username format (is it `LITHOS-2220379` or just `2220379`?)

---

### Issue: "CACCESS Permission Required"

**Meaning**: Credentials work, but account lacks content access permissions

**Solution**:
- Contact FactSet support: misa.kobayashi@factset.com
- Request: "Please enable CACCESS permissions for account LITHOS-2220379"

---

### Issue: "No filings found"

**Possible causes**:
1. Company ticker format is incorrect
   - Should be: `IVN-CA` (not `IVN-TOR` or `IVN.TO`)
2. Date range too narrow
   - Try expanding: change `startDate: '20240101'` to `'20230101'`
3. Company doesn't file with SEDAR
   - Try different source: `'SDR'` for old SEDAR, `'EDG'` for EDGAR (US)

---

## Environment Variables

Required in `.env.local`:

```bash
# FactSet API Credentials
FACTSET_USERNAME=LITHOS-2220379
FACTSET_API_KEY=your-api-key-here
```

---

## File Structure

```
lithos/
├── scripts/
│   ├── verify-factset-credentials.ts       # Test authentication
│   ├── test-factset-download-final.ts      # Main download script
│   ├── test-factset-simple.ts             # Single URL test
│   └── populate-projects-from-factset.ts   # Production script (uses same auth)
├── downloads/
│   └── factset-test/                      # Downloaded documents
└── .env.local                             # Credentials (not in git)
```

---

## Understanding the Scripts

### verify-factset-credentials.ts

**Purpose**: Diagnose authentication issues

**Tests**:
1. `/meta/sources` - List available filing sources
2. `/meta/time-zones` - List time zones
3. `/meta/categories` - List document categories
4. `/search` - Simple filing search

All 4 should pass with valid credentials.

---

### test-factset-download-final.ts

**Purpose**: End-to-end test of search + download

**Flow**:
```
Search for filings
  ↓
Display metadata
  ↓
Download documents
  ↓
Save to disk
  ↓
Show summary
```

**Companies tested**:
- Ivanhoe Mines (`IVN-CA`)
- Teck Resources (`TECK-CA`)
- First Quantum Minerals (`FM-CA`)

**Date range**: 2024-01-01 to present

**Sources**: `SDR,SDRP` (both old and new SEDAR)

---

### test-factset-simple.ts

**Purpose**: Test the specific filing URL from your email

**URL tested**:
```
https://api.factset.com/global-filings/v2/filings?report=story&timezone=America/New_York&key=...
```

This was the URL that gave you the original CACCESS error.

---

## Next Steps After Auth Works

### 1. Verify Downloads Work

Run `test-factset-download-final.ts` and confirm you can:
- ✅ Search for filings
- ✅ See document metadata
- ✅ Download at least one document
- ✅ Open the downloaded PDF/HTML

### 2. Integrate into Production

Once downloads work, update `populate-projects-from-factset.ts`:

```typescript
// Add document download capability
async function downloadFilingDocument(filing: FactSetFilingDocument) {
  const authHeader = createAuthHeader()

  const response = await fetch(filing.filingsLink, {
    headers: {
      'Authorization': authHeader,
      'Accept': 'application/pdf,text/html'
    }
  })

  if (!response.ok) {
    throw new Error(`Download failed: ${await response.text()}`)
  }

  return await response.arrayBuffer()
}
```

### 3. Store Documents

Options:
1. **Local storage**: Save PDFs to `downloads/factset-docs/`
2. **Supabase Storage**: Upload to Supabase bucket
3. **S3/Cloud Storage**: Upload to cloud provider
4. **Extract + Discard**: Parse PDF, extract data, discard file

### 4. Parse Documents

For NI 43-101 technical reports, extract:
- Project name and location
- Commodities and resources
- Financial metrics (NPV, IRR, CAPEX, OPEX)
- Production estimates
- Mine life and timeline

Use OpenAI to parse PDFs:
```typescript
// Parse technical report
const extracted = await openai.chat.completions.create({
  model: "gpt-4",
  messages: [{
    role: "system",
    content: "Extract mining project data from this NI 43-101 technical report"
  }, {
    role: "user",
    content: pdfText
  }]
})
```

---

## API Reference

### FactSet Global Filings API v2

**Base URL**: `https://api.factset.com/content/global-filings/v2`

**Authentication**: Basic Auth
```
Authorization: Basic base64(username:api_key)
```

**Key Endpoints**:
- `GET /search` - Search for filings
- `GET /filings` - Download document content
- `GET /meta/sources` - List available sources
- `GET /meta/form-types` - List form types

**Rate Limits**: 10 requests/second

**Documentation**: https://developer.factset.com/api-catalog/global-filings-api

---

## Support

### FactSet API Issues
- **Contact**: Misa Kobayashi
- **Email**: misa.kobayashi@factset.com
- **Account**: LITHOS-2220379

### Questions to Ask FactSet

If authentication fails:
1. Is the account fully provisioned for API access?
2. Is Basic Auth the correct method?
3. Are there IP restrictions?
4. What is the correct username format?
5. Is the API key active and associated with the account?

If downloads fail with CACCESS:
1. Please enable CACCESS permissions for account LITHOS-2220379
2. Are there different CACCESS levels? Which do we need?
3. How long does it take to provision?

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| 403 on all endpoints | Auth not working | Contact FactSet support |
| 403 on `/filings` only | CACCESS needed | Request CACCESS permissions |
| Empty search results | Wrong ticker format | Use `-CA` suffix for Canadian |
| No recent filings | Date range too narrow | Expand to 2023 or earlier |
| Download timeout | Large file | Increase timeout to 60s |
| PDF corrupt | Encoding issue | Save as binary, not text |

---

**Last Updated**: 2025-10-20
**Status**: Authentication blocked - awaiting FactSet support
